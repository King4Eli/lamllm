"""Download Wikipedia articles and train LamLLM on them.

    python websearch/wiki_train.py --mb 10                       # random articles
    python websearch/wiki_train.py --mb 10 --topics "physics,history,animals"
    python ask.py                                     # then ask questions

Articles come from the official Wikipedia API (plain text, no HTML scraping)
and are cached in data/wiki/articles.jsonl. Re-running reuses the cache and
only downloads more if you ask for a bigger --mb. Besides the articles, the
training text includes Q/A pairs built from each article's opening sentences,
so the model learns to respond to "Q: ...\\nA:" prompts.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lamllm.trainer import train_model  # noqa: E402

API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "LamLLM/0.1 (educational project; local user)"
WIKI_DIR = os.path.join(ROOT, "data", "wiki")
ARTICLES = os.path.join(WIKI_DIR, "articles.jsonl")
TRAIN_TXT = os.path.join(WIKI_DIR, "train.txt")

END_SECTIONS = {
    "references", "see also", "external links", "notes", "further reading",
    "bibliography", "sources", "citations", "footnotes", "notes and references",
}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def api_get(params, retries=5):
    params = {"format": "json", "formatversion": "2", **params}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt
            print(f"  Wikipedia returned {e.code}, waiting {wait}s...", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                raise
            print(f"  request failed ({e}), retrying...", flush=True)
            time.sleep(2 ** attempt)


def to_ascii(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def clean_extract(text):
    """Drop reference sections, flatten headings, keep plain ASCII."""
    out = []
    for line in to_ascii(text).splitlines():
        m = re.match(r"^\s*(=+)\s*(.*?)\s*=+\s*$", line)
        if m:
            heading = m.group(2)
            if heading.lower() in END_SECTIONS:
                break
            out.append("\n" + heading)
        else:
            out.append(line.strip())
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_extract(title):
    data = api_get({"action": "query", "prop": "extracts", "explaintext": 1,
                    "redirects": 1, "titles": title})
    pages = data.get("query", {}).get("pages", [])
    if not pages or "extract" not in pages[0]:
        return None, None
    return pages[0]["title"], pages[0]["extract"]


def random_titles(min_bytes):
    """A batch of random article titles whose source is at least min_bytes long."""
    data = api_get({"action": "query", "generator": "random", "grnnamespace": 0,
                    "grnlimit": 50, "prop": "info"})
    pages = data.get("query", {}).get("pages", [])
    return [p["title"] for p in pages if p.get("length", 0) >= min_bytes]


def topic_titles(topic, limit=50):
    data = api_get({"action": "query", "list": "search", "srsearch": topic,
                    "srlimit": limit, "srnamespace": 0})
    return [r["title"] for r in data.get("query", {}).get("search", [])]


def load_articles():
    if not os.path.exists(ARTICLES):
        return []
    with open(ARTICLES, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def download(target_mb, topics, min_chars, delay):
    os.makedirs(WIKI_DIR, exist_ok=True)
    articles = load_articles()
    seen = {a["title"] for a in articles}
    total = sum(len(a["text"]) for a in articles)
    target = int(target_mb * 1_000_000)
    print(f"cached: {len(articles)} articles, {total / 1e6:.1f} MB (target {target_mb} MB)")

    topic_queue = []
    for t in topics:
        topic_queue += topic_titles(t)
        time.sleep(delay)

    with open(ARTICLES, "a", encoding="utf-8") as f:
        while total < target:
            if topic_queue:
                batch, topic_queue = topic_queue, []
            else:
                batch = random_titles(min_bytes=min_chars * 2)
                time.sleep(delay)
            for title in batch:
                if total >= target:
                    break
                if title in seen:
                    continue
                seen.add(title)
                title, raw = fetch_extract(title)
                time.sleep(delay)
                if not raw or "may refer to" in raw[:300]:
                    continue
                text = clean_extract(raw)
                if len(text) < min_chars:
                    continue
                f.write(json.dumps({"title": to_ascii(title), "text": text}) + "\n")
                f.flush()
                articles.append({"title": to_ascii(title), "text": text})
                total += len(text)
                print(f"  [{total / 1e6:5.2f}/{target_mb} MB] {to_ascii(title)}", flush=True)
    return articles


def subject(title):
    """'Mercury (planet)' -> 'Mercury'."""
    return re.sub(r"\s*\(.*?\)\s*$", "", title)


def qa_pairs(article):
    lead = article["text"].split("\n\n")[0].replace("\n", " ")
    sents = SENTENCE_SPLIT.split(lead)
    if not sents or len(sents[0]) < 20:
        return []
    s = subject(article["title"])
    short = sents[0]
    longer = " ".join(sents[:3])
    return [
        f"Q: What is {s}?\nA: {short}\n",
        f"Q: what is {s.lower()}?\nA: {short}\n",
        f"Q: Who or what is {s}?\nA: {short}\n",
        f"Q: Tell me about {s}.\nA: {longer}\n",
    ]


def build_training_text(articles, qa_repeat, seed):
    chunks = [f"{a['title']}\n\n{a['text']}\n" for a in articles]
    for a in articles:
        chunks += qa_pairs(a) * qa_repeat
    random.Random(seed).shuffle(chunks)
    return "\n".join(chunks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mb", type=float, default=10, help="how much article text to download (MB)")
    p.add_argument("--topics", default="", help="comma-separated topics to fetch first, e.g. 'space,biology'")
    p.add_argument("--min-chars", type=int, default=2000, help="skip articles shorter than this")
    p.add_argument("--delay", type=float, default=0.2, help="seconds between API requests")
    p.add_argument("--contact", default="", help="email or URL added to the User-Agent, as Wikimedia asks")
    p.add_argument("--qa-repeat", type=int, default=3, help="how many times each Q/A pair appears")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "wiki"))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if args.contact:
        global USER_AGENT
        USER_AGENT = f"LamLLM/0.1 (educational project; {args.contact})"

    topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    articles = download(args.mb, topics, args.min_chars, args.delay)

    text = build_training_text(articles, args.qa_repeat, args.seed)
    with open(TRAIN_TXT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"training text: {len(text) / 1e6:.1f} MB from {len(articles)} articles -> {TRAIN_TXT}")
    if args.download_only:
        return

    train_model(
        text, args.out, steps=args.steps, batch_size=args.batch_size, block_size=args.block_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
        lr=args.lr, eval_every=args.eval_every, seed=args.seed,
    )
    print(f'\nDone. Ask questions with:  python ask.py --ckpt {args.out}')


if __name__ == "__main__":
    main()
