"""Ask LamLLM questions in an interactive prompt.

    python ask.py                      # uses checkpoints/wiki and data/wiki
    python ask.py --question "What is the Moon?"

Each answer has two parts:
  Wikipedia  the best-matching passage from the downloaded articles (keyword
             search). This is where reliable facts come from.
  LamLLM     what the model itself generates for "Q: <question>\\nA:".
             A model this small produces fluent-looking but often wrong text.
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

import torch

from lamllm import CharTokenizer, LamConfig, LamLLM
from lamllm.trainer import get_device

ROOT = os.path.dirname(os.path.abspath(__file__))

STOPWORDS = set("""
a an and are as at be by can did do does for from had has have how i in is it its
me of on or tell that the their there this to was were what when where which who
whom whose why will with you your about please explain describe
""".split())
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class Retriever:
    """BM25 search over article paragraphs, with extra weight on title matches."""

    def __init__(self, articles_path, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.passages = []  # (title, paragraph)
        self.lengths = []
        self.postings = defaultdict(list)  # word -> [(passage_id, tf)]
        with open(articles_path, encoding="utf-8") as f:
            for line in f:
                a = json.loads(line)
                title_words = words(a["title"])
                for para in a["text"].split("\n\n"):
                    para = para.strip()
                    if len(para) < 80:
                        continue
                    pid = len(self.passages)
                    self.passages.append((a["title"], para))
                    toks = words(para) + title_words * 3
                    self.lengths.append(len(toks))
                    for w, tf in Counter(toks).items():
                        self.postings[w].append((pid, tf))
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)

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
    ckpt = torch.load(os.path.join(ckpt_dir, "lamllm.pt"), map_location=device)
    tok = CharTokenizer.load(os.path.join(ckpt_dir, "tokenizer.json"))
    model = LamLLM(LamConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
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
    p.add_argument("--articles", default=os.path.join(ROOT, "data", "wiki", "articles.jsonl"))
    p.add_argument("--question", help="ask one question and exit")
    p.add_argument("--max-chars", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--model-only", action="store_true", help="skip the Wikipedia search")
    args = p.parse_args()

    device = get_device()
    model, tok = load_model(args.ckpt, device)

    retriever = None
    if not args.model_only and os.path.exists(args.articles):
        retriever = Retriever(args.articles)
        print(f"indexed {len(retriever.passages):,} passages from {args.articles}")

    def respond(question):
        if retriever:
            title, passage = retriever.answer(question)
            if passage:
                print(f"\nWikipedia ({title}):\n  {passage}")
            else:
                print("\nWikipedia: no matching article downloaded.")
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
