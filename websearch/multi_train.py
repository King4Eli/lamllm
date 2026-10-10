"""Download text from several sources and train LamLLM on it at the same time.

    python websearch/multi_train.py --fresh --hours 8    # new ~30M-param model, train 8 hours
    python websearch/multi_train.py --hours 4           # continue that model for 4 more hours
    python websearch/multi_train.py --fresh --size large --hours 24
    python websearch/multi_train.py --sources wikipedia,stackexchange --mb 50
    python websearch/multi_train.py --sources fineweb-edu,squad,hf:roneneldan/TinyStories,kaggle:owner/dataset

Sources (see websearch/sources.py):
  web APIs       wikipedia, simplewiki (Simple English Wikipedia), stackexchange
                 (questions with top-voted answers), gutenberg (public-domain books)
  Hugging Face   fineweb-edu, fineweb, wikipedia-full, cosmopedia, tinystories, openwebtext,
                 arxiv, pubmed; question/answer sets squad, trivia-qa, dolly, gsm8k;
                 or any dataset as hf:owner/name
  Kaggle         kaggle:owner/name (needs ~/.kaggle/kaggle.json)

Each source downloads in its own background thread while the model trains; new
documents are folded into the training data every few dozen steps, and every batch
mixes text from all sources. Downloads are cached in data/sources/<source>.jsonl, so
later runs reuse them. With --mb 0 (the default) sources keep downloading for as long
as training runs; otherwise each stops at its share of --mb.

By default training continues from the existing model in --out, keeping its size
and tokenizer; use --fresh to start over. A fresh model gets a BPE tokenizer trained
on the downloaded text.
"""

import argparse
import fcntl
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lamllm import BPETokenizer, CharTokenizer  # noqa: E402
from lamllm.trainer import (  # noqa: E402
    DEVICE_CHOICES, Corpus, checkpoint_exists, device_name, fit, get_device, tokenizer_for,
)
from websearch import sources  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "sources")

# name: (n_layer, n_head, n_embd, block_size, batch_size, grad_accum). Batch sizes fit a 4 GB GPU;
# each optimizer step sees batch_size * grad_accum sequences either way.
SIZES = {
    "small": (6, 6, 192, 256, 32, 1),  # ~3M params with the char tokenizer; the original LamLLM
    "medium": (8, 8, 512, 512, 8, 4),  # ~30M params
    "large": (12, 12, 768, 512, 4, 8),  # ~92M params; needs many hours to beat medium
}


DEFAULT_SOURCES = "fineweb-edu,wikipedia-full,cosmopedia,simplewiki,stackexchange,gutenberg,squad,trivia-qa,dolly"


def read_jsonl_tail(path, max_chars):
    """The last ~max_chars of a cached .jsonl file (all of it if smaller), as documents."""
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        f.seek(max(size - int(max_chars), 0))
        if f.tell():
            f.readline()  # skip the partial line we landed in
        docs = []
        for line in f:
            try:
                docs.append(json.loads(line))
            except ValueError:  # a line cut short by a crash
                pass
        return docs


def read_jsonl_head(path, max_chars):
    texts, n = [], 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                texts.append("\n".join([d["text"], *(f"{q}\n{a}" for q, a in d.get("qa", []))]))
                n += len(texts[-1])
                if n >= max_chars:
                    break
    return texts


def doc_key(doc):
    """What makes a document a duplicate: its title, or for untitled ones (titles like '#12'), where it came from."""
    if doc.get("show_title", True):
        return doc["title"]
    return f"{doc.get('source')}|{doc.get('pos')}|{doc['title']}"


def doc_chars(doc):
    return len(doc["text"]) + sum(len(q) + len(a) for q, a in doc.get("qa", []))


def add_document(corpus, doc, qa_repeat):
    # The Q/A pairs stay with their article, so both land in the same train/val split.
    text = f"{doc['title']}\n\n{doc['text']}" if doc.get("show_title", True) else doc["text"]
    parts = [text] if text else []
    parts += [f"Q: {q}\nA: {a}" for q, a in doc.get("qa", []) for _ in range(qa_repeat)]
    corpus.add("\n\n".join(parts), doc.get("source", "text"))


class Downloader(threading.Thread):
    """Fetches documents from one source until it has `target_chars` of text (0 = no limit).

    Pauses while the corpus says it has enough (memory cap, or far ahead of training).
    After an error it waits and carries on from the last document it got, giving up
    only after repeated failures with no progress in between.
    """

    MAX_FAILURES = 6

    def __init__(self, source, corpus, target_chars, have_chars, seen, qa_repeat, last=None):
        super().__init__(daemon=True, name=source.name)
        self.source = source
        self.corpus = corpus
        self.target = target_chars
        self.chars = have_chars
        self.seen = seen
        self.qa_repeat = qa_repeat
        self.last = last
        self.docs = 0
        self.error = None
        self.finished = False  # the source has no more documents
        self.state = "starting"
        self.cache = os.path.join(CACHE_DIR, f"{source.name}.jsonl")

    @property
    def done(self):
        return self.finished or (self.target > 0 and self.chars >= self.target)

    def run(self):
        failures = 0
        while not self.done:
            try:
                self._fetch()
                self.finished = not self.done  # the generator ran out
            except Exception as e:  # one failing source shouldn't stop the others or training
                failures = 0 if self.state == "downloading" else failures
                failures += 1
                if failures > self.MAX_FAILURES:
                    self.error, self.state = e, "error"
                    print(f"\n  [{self.source.name}] stopped after {failures} failures: {e}", flush=True)
                    return
                wait = getattr(e, "seconds", None) or min(30 * 2 ** failures, 1800)
                self.state = f"retry in {wait / 60:.0f}m"
                print(f"\n  [{self.source.name}] {e}; retrying in {wait / 60:.0f} min", flush=True)
                time.sleep(wait)
        self.state = "done"

    def _fetch(self):
        self.state = "connecting"
        with open(self.cache, "a", encoding="utf-8") as f:
            for doc in self.source.documents(self.last):
                self.state = "downloading"
                self.last = doc
                key = doc_key(doc)
                if key in self.seen:
                    continue
                self.seen.add(key)
                f.write(json.dumps(doc) + "\n")
                f.flush()
                add_document(self.corpus, doc, self.qa_repeat)
                self.chars += doc_chars(doc)
                self.docs += 1
                if self.done:
                    return
                while not self.corpus.wants_more():
                    self.state = "paused"
                    time.sleep(5)


def status(downloaders):
    parts = []
    for d in downloaders:
        target = f"/{d.target / 1e6:.0f}" if d.target else ""
        parts.append(f"{d.source.name} {d.chars / 1e6:.1f}{target}MB {d.state}")
    return ", ".join(parts)


def lock_output(out):
    """Stop a second trainer from writing the same checkpoint (and fighting over the GPU)."""
    os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, ".train.lock"), "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.seek(0)
        pid = f.read().strip() or "?"
        raise SystemExit(f"another training run (PID {pid}) is already using {out}. Stop it first "
                         f"(kill {pid}; it saves before exiting), or use --out <dir> for a separate model.")
    f.truncate(0)
    f.write(str(os.getpid()))
    f.flush()
    return f  # keep open: the lock lasts as long as this process


def model_config(ckpt_dir):
    """Size settings of the model saved in `ckpt_dir`."""
    import torch

    return torch.load(os.path.join(ckpt_dir, "lamllm.pt"), map_location="cpu")["config"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default=DEFAULT_SOURCES, help="comma-separated; see the list above")
    p.add_argument("--mb", type=float, default=0,
                   help="total text to download, split evenly between sources; 0 = keep downloading while training")
    p.add_argument("--topics", default="", help="comma-separated topics for the Wikipedia sources to fetch first")
    p.add_argument("--se-sites", default="", help="comma-separated Stack Exchange sites, e.g. 'history,biology'")
    p.add_argument("--se-key", default=os.environ.get("STACKEXCHANGE_KEY"),
                   help="Stack Exchange API key: 10,000 requests a day instead of 300 (or set STACKEXCHANGE_KEY)")
    p.add_argument("--ram-gb", type=float, default=3,
                   help="memory for training text; downloads pause when it's full (2 bytes per token)")
    p.add_argument("--contact", default="", help="email or URL added to the User-Agent, as Wikimedia asks")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between requests to each Wikipedia")
    p.add_argument("--qa-repeat", type=int, default=3, help="how many times each Q/A pair appears")
    p.add_argument("--min-start-mb", type=float, default=2, help="data needed before training starts")
    p.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "wiki"))
    p.add_argument("--device", default="auto", choices=DEVICE_CHOICES,
                   help="auto (GPU if available, else CPU), gpu (require a GPU), cpu, or a specific GPU type")
    p.add_argument("--fresh", action="store_true", help="train a new model instead of continuing the one in --out")
    p.add_argument("--hours", type=float, default=None, help="train for this long (stops at --steps if that comes first)")
    p.add_argument("--steps", type=int, default=None, help="optimizer steps; default 3000 when --hours isn't given")
    p.add_argument("--size", choices=SIZES, default=None, help="new models only: preset size (default medium)")
    p.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe", help="new models only")
    p.add_argument("--vocab-size", type=int, default=8192, help="new models only: BPE vocabulary size")
    p.add_argument("--batch-size", type=int, default=None, help="sequences per forward pass (default from --size)")
    p.add_argument("--grad-accum", type=int, default=None, help="forward passes per optimizer step (default from --size)")
    p.add_argument("--block-size", type=int, default=None, help="new models only: context length in tokens")
    p.add_argument("--n-layer", type=int, default=None, help="new models only")
    p.add_argument("--n-head", type=int, default=None, help="new models only")
    p.add_argument("--n-embd", type=int, default=None, help="new models only")
    p.add_argument("--dropout", type=float, default=0.1, help="new models only")
    p.add_argument("--lr", type=float, default=None, help="default 6e-4 for new models, 3e-4 when continuing")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    lock = lock_output(args.out)  # noqa: F841
    device = get_device(args.device)
    print(f"training on {device_name(device)}")
    if args.contact:
        sources.set_contact(args.contact)
    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    se_sites = [s.strip() for s in args.se_sites.split(",") if s.strip()]
    srcs = sources.make_sources(names, topics, se_sites, args.delay, args.se_key, os.path.join(ROOT, "data", "kaggle"))
    names = [s.name for s in srcs]  # "hf:owner/name" -> "hf_owner_name"

    resume_from = None if args.fresh or not checkpoint_exists(args.out) else args.out
    new_model_flags = [f for f in ("size", "block_size", "n_layer", "n_head", "n_embd") if getattr(args, f) is not None]
    if resume_from and new_model_flags:
        raise SystemExit(f"--{new_model_flags[0].replace('_', '-')} only applies to a new model, but {args.out} "
                         f"already has one, which training would continue. Add --fresh to replace it, "
                         f"or --out <dir> to train the new model somewhere else.")
    n_layer, n_head, n_embd, block_size, batch_size, grad_accum = SIZES[args.size or "medium"]
    if resume_from:
        # The model keeps its own shape; use the batch shape of the preset it matches.
        cfg = model_config(resume_from)
        shape = (cfg["n_layer"], cfg["n_head"], cfg["n_embd"], cfg["block_size"])
        batch_size, grad_accum = next((v[4:] for v in SIZES.values() if v[:4] == shape), (8, 4))
        n_layer = n_head = n_embd = None
        block_size = cfg["block_size"]
    n_layer = args.n_layer or n_layer
    n_head = args.n_head or n_head
    n_embd = args.n_embd or n_embd
    block_size = args.block_size or block_size
    batch_size = args.batch_size or batch_size
    grad_accum = args.grad_accum or grad_accum

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = {name: os.path.join(CACHE_DIR, f"{name}.jsonl") for name in names}
    if resume_from:
        tok = tokenizer_for(resume_from, None)
    elif args.tokenizer == "char":
        tok = CharTokenizer.ascii()
    else:
        # BPE vocabulary from a sample of the cache, or, with too little cached, from the first downloads.
        sample = [t for name in names for t in read_jsonl_head(cache[name], 20e6)]
        tok = None
        if sum(map(len, sample)) >= args.min_start_mb * 1e6:
            print(f"training a {args.vocab_size}-token BPE tokenizer on {sum(map(len, sample)) / 1e6:.0f} MB of cached text...")
            tok = BPETokenizer.train(sample, args.vocab_size)
        del sample
    lr = args.lr or (3e-4 if resume_from else 6e-4)
    max_tokens = int(args.ram_gb * 1e9 / 2)
    corpus = Corpus(tok, max_tokens=max_tokens)

    # Cached documents from earlier runs: the most recent part of each source's cache, up to
    # its share of memory. Downloads then continue after the last cached document.
    seen = set()
    have, last = {}, {}
    for name in names:
        docs = read_jsonl_tail(cache[name], max_tokens * 3.5 / len(names))
        for d in docs:
            if doc_key(d) not in seen:
                seen.add(doc_key(d))
                add_document(corpus, d, args.qa_repeat)
        corpus.refresh()  # tokenize source by source to keep memory down
        have[name] = os.path.getsize(cache[name]) if os.path.exists(cache[name]) else 0
        last[name] = docs[-1] if docs else None
        if docs:
            print(f"cached {name}: {have[name] / 1e6:,.0f} MB on disk, using the last {len(docs):,} documents")

    share = int(args.mb * 1_000_000 / len(srcs))
    downloaders = [Downloader(s, corpus, share, have[s.name], seen, args.qa_repeat, last[s.name]) for s in srcs]
    for d in downloaders:
        d.start()

    def wait(condition):
        while not condition():
            if all(not d.is_alive() for d in downloaders):
                raise SystemExit("downloads stopped before there was enough data to train on")
            print(f"\r  waiting for data: {status(downloaders)}".ljust(120), end="", flush=True)
            time.sleep(2)
            corpus.refresh()
        print()

    # Wait for enough data to start training (and to train the tokenizer on).
    wait(lambda: corpus.chars >= args.min_start_mb * 1e6)
    if corpus.tok is None:
        print(f"training a {args.vocab_size}-token BPE tokenizer on {corpus.chars / 1e6:.1f} MB of downloaded text...")
        corpus.set_tokenizer(BPETokenizer.train(corpus.pending_texts(), args.vocab_size))
    corpus.refresh()
    wait(lambda: corpus.ready(block_size))

    fit(corpus, args.out, steps=args.steps, hours=args.hours, batch_size=batch_size, grad_accum=grad_accum,
        block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=args.dropout,
        lr=lr, eval_every=args.eval_every, seed=args.seed, resume_from=resume_from, device=device)

    print(f"downloads: {status(downloaders)}")
    if any(not d.done for d in downloaders):
        print("Unfinished downloads are cached; the next run picks up where they left off.")
    print("Ask questions with:  python ask.py")


if __name__ == "__main__":
    main()
