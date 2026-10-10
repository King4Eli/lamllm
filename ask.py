"""Ask LamLLM questions in an interactive prompt.

    python ask.py                      # uses checkpoints/wiki and everything downloaded in data/
    python ask.py --question "What is the Moon?"

Each answer has two parts:
  Source     the best-matching passage from the downloaded Wikipedia articles and
             Stack Exchange answers (keyword search). This is where reliable facts
             come from. Gutenberg books are left out of the search, since they are
             fiction and literature rather than facts.
  LamLLM     what the model itself generates for "Q: <question>\\nA:".
             A model this small produces fluent-looking but often wrong text.
"""

import argparse
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict

import torch

from lamllm.trainer import get_device, load_checkpoint

ROOT = os.path.dirname(os.path.abspath(__file__))
SEARCH_EXCLUDE = {"gutenberg"}

STOPWORDS = set("""
a an and are as at be by can did do does for from had has have how i in is it its
me of on or tell that the their there this to was were what when where which who
whom whose why will with you your about please explain describe
""".split())
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class Retriever:
    """BM25 search over document paragraphs, with extra weight on title matches."""

    def __init__(self, paths, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.passages = []  # (label, paragraph)
        self.lengths = []
        self.postings = defaultdict(list)  # word -> [(passage_id, tf)]
        for path in paths:
            self._index_file(path)
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)

    def _index_file(self, path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                a = json.loads(line)
                source = a.get("source", "wikipedia")
                if source in SEARCH_EXCLUDE:
                    continue
                title_words = words(a["title"])
                for para in a["text"].split("\n\n"):
                    para = para.strip()
                    if len(para) < 80:
                        continue
                    pid = len(self.passages)
                    self.passages.append((f"{source}: {a['title']}", para))
                    toks = words(para) + title_words * 3
                    self.lengths.append(len(toks))
                    for w, tf in Counter(toks).items():
                        self.postings[w].append((pid, tf))

    def search(self, query):
        n = len(self.passages)
        scores = defaultdict(float)
        for w in set(words(query)):
            plist = self.postings.get(w)
            if not plist:
                continue
            idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for pid, tf in plist:
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[pid] / self.avg_len))
                scores[pid] += idf * norm
        if not scores:
            return None
        best = max(scores, key=scores.get)
        return self.passages[best]

    def answer(self, query, max_sentences=3):
        hit = self.search(query)
        if hit is None:
            return None, None
        title, para = hit
        sents = SENTENCE_SPLIT.split(para.replace("\n", " "))
        q = set(words(query))
        start = max(range(len(sents)), key=lambda i: (len(q & set(words(sents[i]))), -i))
        return title, " ".join(sents[start:start + max_sentences])


def load_model(ckpt_dir, device):
    model, tok, _ = load_checkpoint(ckpt_dir, device)
    model.eval()
    return model, tok


def model_answer(model, tok, question, device, max_chars, temperature, top_k):
    prompt = f"Q: {question.strip()}\nA:"
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_chars, temperature=temperature, top_k=top_k)
    text = tok.decode(out[0, idx.size(1):].tolist())
    return text.split("\n")[0].strip()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "wiki"))
    p.add_argument("--data", nargs="*", help="downloaded .jsonl files to search (default: all under data/)")
    p.add_argument("--question", help="ask one question and exit")
    p.add_argument("--max-chars", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--model-only", action="store_true", help="skip the search of downloaded text")
    args = p.parse_args()

    device = get_device()
    model, tok = load_model(args.ckpt, device)

    retriever = None
    paths = args.data if args.data is not None else sorted(glob.glob(os.path.join(ROOT, "data", "*", "*.jsonl")))
    if not args.model_only and paths:
        retriever = Retriever(paths)
        print(f"indexed {len(retriever.passages):,} passages from {len(paths)} file(s)")

    def respond(question):
        if retriever:
            title, passage = retriever.answer(question)
            if passage:
                print(f"\nSource ({title}):\n  {passage}")
            else:
                print("\nSource: nothing matching has been downloaded.")
        reply = model_answer(model, tok, question, device, args.max_chars, args.temperature, args.top_k)
        print(f"\nLamLLM:\n  {reply}\n")

    if args.question:
        respond(args.question)
        return

    print("Ask a question (Ctrl+D or 'quit' to exit).")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            respond(q)


if __name__ == "__main__":
    main()
