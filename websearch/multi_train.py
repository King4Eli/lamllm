"""Download text from several sources and train LamLLM on it at the same time.

    python websearch/multi_train.py                     # all sources, 20 MB in total
    python websearch/multi_train.py --sources wikipedia,stackexchange --mb 10
    python websearch/multi_train.py --fresh             # start a new model instead of continuing

Sources: wikipedia, simplewiki (Simple English Wikipedia), stackexchange (questions
with top-voted answers), gutenberg (public-domain books).

Each source downloads in its own background thread while the model trains; new
documents are folded into the training data every few dozen steps. Downloads are
cached in data/sources/<source>.jsonl, so later runs reuse them and only fetch
more until each source reaches its share of --mb.

By default training continues from the existing model in --out, keeping its size
and character set; use --fresh to start over.
"""

import argparse
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lamllm import CharTokenizer  # noqa: E402
from lamllm.trainer import Corpus, checkpoint_exists, fit, tokenizer_for  # noqa: E402
from websearch import sources  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "sources")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def add_document(corpus, doc, qa_repeat):
    corpus.add(f"{doc['title']}\n\n{doc['text']}")
    for q, a in doc.get("qa", []):
        for _ in range(qa_repeat):
            corpus.add(f"Q: {q}\nA: {a}")


class Downloader(threading.Thread):
    """Fetches documents from one source until it has `target_chars` of text."""

    def __init__(self, source, corpus, target_chars, have_chars, seen, qa_repeat):
        super().__init__(daemon=True, name=source.name)
        self.source = source
        self.corpus = corpus
        self.target = target_chars
        self.chars = have_chars
        self.seen = seen
        self.qa_repeat = qa_repeat
        self.docs = 0
        self.error = None
        self.cache = os.path.join(CACHE_DIR, f"{source.name}.jsonl")

    @property
    def done(self):
        return self.chars >= self.target

    def run(self):
        if self.done:
            return
        try:
            with open(self.cache, "a", encoding="utf-8") as f:
                for doc in self.source.documents():
                    if doc["title"] in self.seen:
                        continue
                    self.seen.add(doc["title"])
                    f.write(json.dumps(doc) + "\n")
                    f.flush()
                    add_document(self.corpus, doc, self.qa_repeat)
                    self.chars += len(doc["text"])
                    self.docs += 1
                    if self.done:
                        break
        except Exception as e:  # one failing source shouldn't stop the others or training
            self.error = e
            print(f"\n  [{self.source.name}] stopped: {e}", flush=True)


def status(downloaders):
    parts = []
    for d in downloaders:
        state = "error" if d.error else "done" if d.done else "downloading"
        parts.append(f"{d.source.name} {d.chars / 1e6:.2f}/{d.target / 1e6:.1f} MB ({state})")
    return ", ".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", default="wikipedia,simplewiki,stackexchange,gutenberg")
    p.add_argument("--mb", type=float, default=20, help="total text to download, split evenly between sources")
    p.add_argument("--topics", default="", help="comma-separated topics for the Wikipedia sources to fetch first")
    p.add_argument("--se-sites", default="", help="comma-separated Stack Exchange sites, e.g. 'history,biology'")
    p.add_argument("--contact", default="", help="email or URL added to the User-Agent, as Wikimedia asks")
    p.add_argument("--delay", type=float, default=0.5, help="seconds between requests to each Wikipedia")
    p.add_argument("--qa-repeat", type=int, default=3, help="how many times each Q/A pair appears")
    p.add_argument("--min-start-mb", type=float, default=0.3, help="data needed before training starts")
    p.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "wiki"))
    p.add_argument("--fresh", action="store_true", help="train a new model instead of continuing the one in --out")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256, help="new models only")
    p.add_argument("--n-layer", type=int, default=6, help="new models only")
    p.add_argument("--n-head", type=int, default=6, help="new models only")
    p.add_argument("--n-embd", type=int, default=192, help="new models only")
    p.add_argument("--dropout", type=float, default=0.1, help="new models only")
    p.add_argument("--lr", type=float, default=None, help="default 1e-3 for new models, 3e-4 when continuing")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if args.contact:
        sources.set_contact(args.contact)
    names = [n.strip() for n in args.sources.split(",") if n.strip()]
    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    se_sites = [s.strip() for s in args.se_sites.split(",") if s.strip()]
    srcs = sources.make_sources(names, topics, se_sites, args.delay)

    resume_from = None if args.fresh or not checkpoint_exists(args.out) else args.out
    tok = tokenizer_for(resume_from, CharTokenizer.ascii())
    lr = args.lr or (3e-4 if resume_from else 1e-3)
    corpus = Corpus(tok)

    # Cached documents from earlier runs.
    os.makedirs(CACHE_DIR, exist_ok=True)
    seen = set()
    have = {}
    for name in names:
        docs = read_jsonl(os.path.join(CACHE_DIR, f"{name}.jsonl"))
        for d in docs:
            if d["title"] not in seen:
                seen.add(d["title"])
                add_document(corpus, d, args.qa_repeat)
        have[name] = sum(len(d["text"]) for d in docs)
        print(f"cached {name}: {len(docs)} documents, {have[name] / 1e6:.2f} MB")

    share = int(args.mb * 1_000_000 / len(srcs))
    downloaders = [Downloader(s, corpus, share, have[s.name], seen, args.qa_repeat) for s in srcs]
    for d in downloaders:
        d.start()

    # Wait for enough data to start training.
    corpus.refresh()
    while corpus.size("train") + corpus.size("val") < args.min_start_mb * 1e6 or corpus.size("val") < 20_000:
        if all(not d.is_alive() for d in downloaders):
            raise SystemExit("downloads stopped before there was enough data to train on")
        print(f"\r  waiting for data: {status(downloaders)}".ljust(120), end="", flush=True)
        time.sleep(2)
        corpus.refresh()
    print()

    fit(corpus, args.out, steps=args.steps, batch_size=args.batch_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
        lr=lr, eval_every=args.eval_every, seed=args.seed, resume_from=resume_from)

    print(f"downloads: {status(downloaders)}")
    if any(not d.done and not d.error for d in downloaders):
        print("Unfinished downloads are cached; the next run picks up where they left off.")
    print("Ask questions with:  python ask.py")


if __name__ == "__main__":
    main()
