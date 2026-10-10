"""Text sources LamLLM can learn from.

Each source's documents(last) is a generator yielding dicts:
    {"source": ..., "title": ..., "text": ..., "qa": [[question, answer], ...], "pos": ...}
Text is cleaned to plain ASCII. "qa" holds question/answer pairs the model is
also trained on, so it learns to respond to "Q: ...\\nA:" prompts. "pos" marks how
far through the source a document is; `last` is the last document cached by an
earlier run, so a source can carry on where it stopped instead of starting over.
Documents with "show_title": False are trained on without their title (used when
the title is just an id or URL).

Kinds of source:
  web APIs        Wikipedia, Stack Exchange, Gutenberg: one request at a time, with
                  a delay, backing off when a site says it is overloaded.
  Hugging Face    datasets streamed from the Hub's Parquet files a row group at a
                  time, so even terabyte-sized datasets start at once. "hf:owner/name"
                  works for any dataset; the presets below pick the right columns.
  Kaggle          "kaggle:owner/name" downloads a dataset (needs a Kaggle API key)
                  and reads text out of its .txt/.csv/.json/.jsonl/.parquet files.
"""

import base64
import csv
import gzip
import html
import io
import json
import os
import re
import time
import zipfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

USER_AGENT = "LamLLM/0.1 (educational project; local user)"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def set_contact(contact):
    """Wikimedia asks automated clients to include contact details in the User-Agent."""
    global USER_AGENT
    USER_AGENT = f"LamLLM/0.1 (educational project; {contact})"


def http_get(url, params=None, retries=5, timeout=60):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return body
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt
            print(f"  [{urllib.parse.urlparse(url).netloc}] returned {e.code}, waiting {wait}s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def get_json(url, params=None):
    return json.loads(http_get(url, params))


def to_ascii(s):
    """Plain ASCII: accents stripped, tabs to spaces, other control characters dropped."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace("\r\n", "\n").replace("\t", " ")
    return "".join(c for c in s if c == "\n" or 32 <= ord(c) < 127)


def sentences(text, n):
    return " ".join(SENTENCE_SPLIT.split(text.replace("\n", " ").strip())[:n])


def subject(title):
    """'Mercury (planet)' -> 'Mercury'."""
    return re.sub(r"\s*\(.*?\)\s*$", "", title)


# --- Wikipedia (any language edition, e.g. simple.wikipedia.org) ---------------

END_SECTIONS = {
    "references", "see also", "external links", "notes", "further reading",
    "bibliography", "sources", "citations", "footnotes", "notes and references",
    "related pages", "other websites",
}


def clean_wiki_extract(text):
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
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def wiki_qa(title, text):
    lead = text.split("\n\n")[0]
    short = sentences(lead, 1)
    if len(short) < 20:
        return []
    s = subject(title)
    return [
        [f"What is {s}?", short],
        [f"what is {s.lower()}?", short],
        [f"Who or what is {s}?", short],
        [f"Tell me about {s}.", sentences(lead, 3)],
    ]


class Wikipedia:
    def __init__(self, name="wikipedia", host="en.wikipedia.org", min_chars=2000, topics=(), delay=0.2):
        self.name = name
        self.api = f"https://{host}/w/api.php"
        self.min_chars = min_chars
        self.topics = list(topics)
        self.delay = delay

    def _query(self, **params):
        time.sleep(self.delay)
        return get_json(self.api, {"action": "query", "format": "json", "formatversion": "2", **params})

    def _titles(self):
        for topic in self.topics:
            data = self._query(list="search", srsearch=topic, srlimit=50, srnamespace=0)
            yield from (r["title"] for r in data.get("query", {}).get("search", []))
        while True:
            data = self._query(generator="random", grnnamespace=0, grnlimit=50, prop="info")
            for p in data.get("query", {}).get("pages", []):
                if p.get("length", 0) >= self.min_chars * 2:
                    yield p["title"]

    def documents(self, last=None):
        for title in self._titles():
            data = self._query(prop="extracts", explaintext=1, redirects=1, titles=title)
            pages = data.get("query", {}).get("pages", [])
            if not pages or "extract" not in pages[0]:
                continue
            raw = pages[0]["extract"]
            if "may refer to" in raw[:300]:
                continue
            text = clean_wiki_extract(raw)
            if len(text) < self.min_chars:
                continue
            title = to_ascii(pages[0]["title"])
            yield {"source": self.name, "title": title, "text": text, "qa": wiki_qa(title, text)}


# --- Stack Exchange: real questions with their top-voted answers --------------

class _HTMLText(HTMLParser):
    BLOCKS = {"p", "br", "li", "pre", "blockquote", "h1", "h2", "h3", "h4", "tr"}

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def html_to_text(s):
    parser = _HTMLText()
    parser.feed(s)
    text = to_ascii("".join(parser.parts))
    text = re.sub(r"[ ]{2,}", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


class RetryLater(Exception):
    """A source is temporarily out of quota; the downloader should wait `seconds` and resume."""

    def __init__(self, message, seconds):
        super().__init__(message)
        self.seconds = seconds


class StackExchange:
    """Without an API key Stack Exchange allows ~300 requests a day per IP; with a free key
    (register an app at https://stackapps.com/apps/oauth/register) it's 10,000."""

    API = "https://api.stackexchange.com/2.3"

    def __init__(self, sites=("history", "biology", "physics", "astronomy", "cooking", "english", "travel", "diy"),
                 delay=1.0, key=None):
        self.name = "stackexchange"
        self.sites = list(sites)
        self.delay = delay
        self.key = key

    def _get(self, path, **params):
        time.sleep(self.delay)
        if self.key:
            params["key"] = self.key
        try:
            data = get_json(f"{self.API}/{path}", params)
        except urllib.error.HTTPError as e:
            body = e.read()
            try:
                err = json.loads(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body)
            except ValueError:
                raise e
            if err.get("error_name") == "throttle_violation":
                m = re.search(r"(\d+) seconds", err.get("error_message", ""))
                wait = int(m.group(1)) + 60 if m else 3600
                raise RetryLater(f"daily request quota used up (add --se-key for more); "
                                 f"resuming in {wait / 3600:.1f} hours", wait)
            raise RuntimeError(f"{e.code} {err.get('error_name')}: {err.get('error_message')}")
        if data.get("backoff"):  # the API asks clients to pause when told to
            time.sleep(data["backoff"])
        if data.get("quota_remaining") == 0:
            raise RetryLater("daily request quota used up; resuming tomorrow", 24 * 3600)
        return data

    def documents(self, last=None):
        page = (last or {}).get("pos") or 1
        while True:
            for site in self.sites:
                qs = self._get("questions", site=site, order="desc", sort="votes", page=page,
                               pagesize=50, filter="withbody").get("items", [])
                answered = [q for q in qs if q.get("is_answered") and q.get("answer_count")]
                if not answered:
                    continue
                ids = ";".join(str(q["question_id"]) for q in answered)
                answers = self._get(f"questions/{ids}/answers", site=site, order="desc", sort="votes",
                                    pagesize=100, filter="withbody").get("items", [])
                best = {}
                for a in answers:  # sorted by votes, so the first per question is the best
                    best.setdefault(a["question_id"], a)
                for q in answered:
                    a = best.get(q["question_id"])
                    if not a:
                        continue
                    title = to_ascii(html.unescape(q["title"]))
                    answer = html_to_text(a["body"])
                    text = f"Question: {title}\n{html_to_text(q['body'])}\n\nAnswer:\n{answer}"
                    short = sentences(answer, 3)[:600]
                    yield {"source": self.name, "title": f"{title} ({site})", "text": text,
                           "qa": [[title, short]] if len(short) > 20 else [], "pos": page}
            page += 1


# --- Project Gutenberg books ---------------------------------------------------

class Gutenberg:
    """Popular English books. The catalog comes from Gutendex; the text is downloaded
    from Gutenberg's mirror, since gutenberg.org asks bots not to use the main site."""

    CATALOG = "https://gutendex.com/books/"
    MIRROR = "https://aleph.pglaf.org"

    def __init__(self, max_chars=400_000, delay=2.0):
        self.name = "gutenberg"
        self.max_chars = max_chars
        self.delay = delay

    def _mirror_urls(self, book_id):
        s = str(book_id)
        folder = "/".join(s[:-1]) + "/" + s if len(s) > 1 else "0/" + s
        return [f"{self.MIRROR}/{folder}/{s}-0.txt", f"{self.MIRROR}/{folder}/{s}.txt", f"{self.MIRROR}/{folder}/{s}-8.txt"]

    def _download(self, book_id):
        for url in self._mirror_urls(book_id):
            try:
                time.sleep(self.delay)
                return http_get(url).decode("utf-8", errors="ignore")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
        return None

    @staticmethod
    def _strip_boilerplate(text):
        start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
        end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
        return text[start.end() if start else 0:end.start() if end else len(text)]

    def documents(self, last=None):
        url = (last or {}).get("pos") or f"{self.CATALOG}?languages=en"
        while url:
            time.sleep(self.delay)
            data = get_json(url)
            for book in data.get("results", []):
                raw = self._download(book["id"])
                if not raw:
                    continue
                text = to_ascii(self._strip_boilerplate(raw))
                # Books wrap lines at ~70 chars; join wrapped lines back into paragraphs.
                text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()[:self.max_chars]
                if len(text) < 10_000:
                    continue
                authors = ", ".join(a["name"] for a in book.get("authors", []))
                title = to_ascii(book["title"] + (f" by {authors}" if authors else ""))
                yield {"source": self.name, "title": title, "text": text, "qa": [], "pos": url}
            url = data.get("next")


# --- Text pulled out of arbitrary tables (generic Hugging Face and Kaggle data) ---

def text_columns(rows, min_avg=100):
    """Columns whose values are mostly long strings: the ones worth training on."""
    lengths = {}
    for r in rows[:200]:
        for k, v in r.items():
            if isinstance(v, str):
                lengths.setdefault(k, []).append(len(v))
    cols = [k for k, ls in lengths.items() if sum(ls) / len(rows[:200]) >= min_avg]
    return sorted(cols, key=lambda k: -sum(lengths[k]))


def title_column(rows):
    if not rows:
        return None
    by_lower = {k.lower(): k for k, v in rows[0].items() if isinstance(v, str)}
    return next((by_lower[k] for k in ("title", "name", "headline", "subject", "question") if k in by_lower), None)


def table_doc(row, cols, title_col, fallback_title, min_chars=200):
    text = to_ascii("\n\n".join(row[c] for c in cols if isinstance(row.get(c), str) and row[c].strip())).strip()
    if len(text) < min_chars:
        return None
    title = to_ascii(str(row.get(title_col) or "")).strip() if title_col else ""
    return {"title": title or fallback_title, "text": text, "qa": [], "show_title": bool(title)}


# --- Hugging Face datasets ---------------------------------------------------

def clean_plain_wiki(text):
    """Wikipedia dump text (headings as plain lines): cut the reference sections, ASCII only."""
    out = []
    for line in to_ascii(text).splitlines():
        if line.strip().lower() in END_SECTIONS:
            break
        out.append(line.strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


class HuggingFace:
    """Streams a dataset's Parquet files from the Hugging Face Hub, one row group at a time.

    `pattern` is an HfFileSystem glob. `rows_to_docs(rows, start)` turns a list of rows
    into (row_index, doc) pairs; without it, the longest text columns are used. Set
    HF_TOKEN in the environment for gated datasets and higher rate limits.
    """

    def __init__(self, name, pattern, columns=None, rows_to_docs=None):
        self.name = name
        self.pattern = pattern
        self.columns = columns
        self.rows_to_docs = rows_to_docs or self._auto_docs

    def _auto_docs(self, rows, start):
        if not hasattr(self, "_cols"):
            self._cols = text_columns(rows)
            self._title = title_column(rows)
            if not self._cols:
                raise RuntimeError(f"no text columns found in {self.pattern}; columns: {list(rows[0])}")
            print(f"  [{self.name}] training on column(s) {', '.join(self._cols)}", flush=True)
        for i, r in enumerate(rows):
            doc = table_doc(r, self._cols, self._title, f"{self.name} #{start + i}")
            if doc:
                yield start + i, doc

    def documents(self, last=None):
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        files = sorted(fs.glob(self.pattern))
        if not files:
            raise RuntimeError(f"no Parquet files match {self.pattern}")
        first_file, first_row = (last or {}).get("pos") or (0, 0)
        for f in range(first_file, len(files)):
            start = first_row if f == first_file else 0
            pf = pq.ParquetFile(fs.open(files[f], block_size=8 << 20))
            offset = 0
            for g in range(pf.num_row_groups):
                n = pf.metadata.row_group(g).num_rows
                if offset + n > start:  # skip row groups an earlier run already used, without downloading them
                    rows = pf.read_row_group(g, columns=self.columns).to_pylist()
                    skip = max(start - offset, 0)
                    for row, doc in self.rows_to_docs(rows[skip:], offset + skip):
                        doc.update(source=self.name, pos=[f, row + 1])
                        doc.setdefault("title", f"{self.name} #{f}/{row}")
                        yield doc
                offset += n


def _texts(field, title=None, min_chars=200, show_title=False):
    """rows_to_docs for datasets that are one long text column (plus maybe a title)."""
    def convert(rows, start):
        for i, r in enumerate(rows):
            text = to_ascii(r[field] or "").strip()
            if len(text) >= min_chars:
                t = to_ascii(r[title]).strip() if title and r.get(title) else None
                yield start + i, {"title": t or f"#{start + i}", "text": text, "qa": [],
                                  "show_title": show_title and bool(t)}
    return convert


def _wikipedia_rows(rows, start):
    for i, r in enumerate(rows):
        text = clean_plain_wiki(r["text"])
        if len(text) < 1500 or "may refer to" in text[:300]:
            continue
        title = to_ascii(r["title"])
        yield start + i, {"title": title, "text": text, "qa": wiki_qa(title, text)}


def _arxiv_rows(rows, start):
    for i, r in enumerate(rows):
        title = to_ascii(" ".join(r["title"].split()))
        text = to_ascii(" ".join(r["abstract"].split()))
        if len(text) > 200:
            yield start + i, {"title": title, "text": text, "qa": [[f"What is the paper '{title}' about?", sentences(text, 2)]]}


def _pubmed_rows(rows, start):
    for i, r in enumerate(rows):
        text = to_ascii(r["article"]).strip()
        summary = to_ascii(r["abstract"]).strip()
        if len(text) > 500:
            yield start + i, {"title": f"pubmed #{start + i}", "text": f"Abstract: {summary}\n\n{text}", "qa": [],
                              "show_title": False}


def _squad_rows(rows, start):
    """Consecutive rows share a paragraph; each paragraph becomes one document with its questions."""
    doc = None
    for i, r in enumerate(rows):
        context = to_ascii(r["context"]).strip()
        if doc is None or doc["text"] != context:
            if doc:
                yield start + i - 1, doc
            title = to_ascii(r["title"].replace("_", " "))
            doc = {"title": f"{title} #{start + i}", "text": context, "qa": [], "show_title": False}
        if r["answers"]["text"]:
            doc["qa"].append([to_ascii(r["question"]).strip(), to_ascii(r["answers"]["text"][0])])
    if doc:
        yield start + len(rows) - 1, doc


def _qa_rows(question, answer, context=None, min_chars=1):
    """rows_to_docs for question/answer datasets: the text is just the context (if any); the pair goes in "qa"."""
    def convert(rows, start):
        for i, r in enumerate(rows):
            q = to_ascii(r[question]).strip()
            a = answer(r) if callable(answer) else r[answer]
            a = to_ascii(a or "").strip()
            if len(a) < min_chars or not q:
                continue
            ctx = to_ascii(r.get(context) or "").strip() if context else ""
            yield start + i, {"title": q[:200], "text": ctx, "qa": [[q, a]], "show_title": False}
    return convert


def hf_presets():
    return {
        "fineweb-edu": lambda: HuggingFace("fineweb-edu", "datasets/HuggingFaceFW/fineweb-edu/sample/10BT/*.parquet",
                                           ["text"], _texts("text")),
        "fineweb": lambda: HuggingFace("fineweb", "datasets/HuggingFaceFW/fineweb/sample/10BT/*.parquet",
                                       ["text"], _texts("text")),
        "wikipedia-full": lambda: HuggingFace("wikipedia-full", "datasets/wikimedia/wikipedia/20231101.en/*.parquet",
                                              ["title", "text"], _wikipedia_rows),
        "cosmopedia": lambda: HuggingFace("cosmopedia", "datasets/HuggingFaceTB/smollm-corpus/cosmopedia-v2/*.parquet",
                                          ["text"], _texts("text")),
        "tinystories": lambda: HuggingFace("tinystories", "datasets/roneneldan/TinyStories@~parquet/default/train/*.parquet",
                                           ["text"], _texts("text", min_chars=100)),
        "openwebtext": lambda: HuggingFace("openwebtext", "datasets/Skylion007/openwebtext@~parquet/plain_text/train/*.parquet",
                                           ["text"], _texts("text")),
        "arxiv": lambda: HuggingFace("arxiv", "datasets/gfissore/arxiv-abstracts-2021@~parquet/default/train/*.parquet",
                                     ["title", "abstract"], _arxiv_rows),
        "pubmed": lambda: HuggingFace("pubmed", "datasets/ccdv/pubmed-summarization@~parquet/document/train/*.parquet",
                                      ["article", "abstract"], _pubmed_rows),
        "squad": lambda: HuggingFace("squad", "datasets/rajpurkar/squad/plain_text/train-*.parquet",
                                     ["title", "context", "question", "answers"], _squad_rows),
        "trivia-qa": lambda: HuggingFace("trivia-qa", "datasets/mandarjoshi/trivia_qa/rc.nocontext/train-*.parquet",
                                         ["question", "answer"], _qa_rows("question", lambda r: r["answer"]["value"])),
        "dolly": lambda: HuggingFace("dolly", "datasets/databricks/databricks-dolly-15k@~parquet/default/train/*.parquet",
                                     ["instruction", "context", "response"],
                                     _qa_rows("instruction", "response", context="context", min_chars=2)),
        "gsm8k": lambda: HuggingFace("gsm8k", "datasets/openai/gsm8k/main/train-*.parquet",
                                     ["question", "answer"], _qa_rows("question", "answer")),
    }


def hf_dataset(repo):
    """Any Hugging Face dataset, via the Hub's automatic Parquet conversion (first config, train split)."""
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    for pattern in (f"datasets/{repo}@~parquet/*/train/*.parquet", f"datasets/{repo}@~parquet/*/*/*.parquet",
                    f"datasets/{repo}/**/*.parquet"):
        files = sorted(fs.glob(pattern))
        if files:
            folder = files[0].rsplit("/", 1)[0]
            return HuggingFace(source_file_name(f"hf:{repo}"), f"{folder}/*.parquet")
    raise SystemExit(f"hf:{repo}: no Parquet files found (check the name at https://huggingface.co/datasets/{repo})")


# --- Kaggle ------------------------------------------------------------------

def kaggle_credentials():
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    path = os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle")), "kaggle.json")
    if not (user and key) and os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        user, key = cfg.get("username"), cfg.get("key")
    if not (user and key):
        raise SystemExit("Kaggle sources need an API key: on kaggle.com go to Settings > API > Create New Token "
                         "and save the downloaded kaggle.json to ~/.kaggle/kaggle.json "
                         "(or set KAGGLE_USERNAME and KAGGLE_KEY).")
    return user, key


class Kaggle:
    """A Kaggle dataset ("owner/name"), downloaded once to data/kaggle/ and read file by file."""

    API = "https://www.kaggle.com/api/v1/datasets/download"
    TEXT_EXT = (".txt", ".md")
    TABLE_EXT = (".csv", ".tsv", ".json", ".jsonl", ".parquet")

    def __init__(self, dataset, download_dir):
        self.dataset = dataset
        self.name = source_file_name(f"kaggle:{dataset}")
        self.zip_path = os.path.join(download_dir, self.name + ".zip")
        self.credentials = kaggle_credentials()

    def _download(self):
        if os.path.exists(self.zip_path):
            return
        os.makedirs(os.path.dirname(self.zip_path), exist_ok=True)
        token = base64.b64encode(":".join(self.credentials).encode()).decode()
        req = urllib.request.Request(f"{self.API}/{self.dataset}",
                                     headers={"User-Agent": USER_AGENT, "Authorization": f"Basic {token}"})
        print(f"  [{self.name}] downloading {self.dataset} from Kaggle...", flush=True)
        tmp = self.zip_path + ".part"
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(tmp, self.zip_path)

    @staticmethod
    def _rows(zf, member):
        lower = member.lower()
        with zf.open(member) as raw:
            if lower.endswith(".parquet"):
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(io.BytesIO(raw.read()))
                for g in range(pf.num_row_groups):
                    yield from pf.read_row_group(g).to_pylist()
                return
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")
            if lower.endswith((".csv", ".tsv")):
                csv.field_size_limit(1 << 30)
                yield from csv.DictReader(text, delimiter="\t" if lower.endswith(".tsv") else ",")
            elif lower.endswith(".jsonl"):
                yield from (json.loads(line) for line in text if line.strip())
            else:
                data = json.load(text)
                if isinstance(data, dict):  # {"data": [...]} and similar wrappers
                    data = next((v for v in data.values() if isinstance(v, list)), [data])
                yield from (d for d in data if isinstance(d, dict))

    def documents(self, last=None):
        self._download()
        first_file, first_row = (last or {}).get("pos") or (0, 0)
        with zipfile.ZipFile(self.zip_path) as zf:
            members = sorted(m for m in zf.namelist() if m.lower().endswith(self.TEXT_EXT + self.TABLE_EXT))
            if not members:
                raise RuntimeError(f"{self.dataset} has no .txt/.md/.csv/.tsv/.json/.jsonl/.parquet files")
            for f in range(first_file, len(members)):
                member = members[f]
                start = first_row if f == first_file else 0
                if member.lower().endswith(self.TEXT_EXT):
                    text = re.sub(r"\n{3,}", "\n\n", to_ascii(zf.read(member).decode("utf-8", "ignore"))).strip()
                    for i in range(start, len(text), 200_000):  # big files become several documents
                        yield {"title": f"{member} part {i // 200_000 + 1}", "text": text[i:i + 200_000], "qa": [],
                               "show_title": False, "source": self.name, "pos": [f, i + 200_000]}
                    continue
                head, cols, title_col = [], None, None
                for i, row in enumerate(self._rows(zf, member)):
                    if cols is None:
                        head.append(row)
                        if len(head) < 200:
                            continue
                        cols, title_col = text_columns(head), title_column(head)
                        pending, head = list(enumerate(head)), None
                    else:
                        pending = [(i, row)]
                    for j, r in pending:
                        if j >= start and cols:
                            doc = table_doc(r, cols, title_col, f"{member} #{j}")
                            if doc:
                                yield {**doc, "source": self.name, "pos": [f, j + 1]}
                if head:  # fewer than 200 rows in the file
                    cols, title_col = text_columns(head), title_column(head)
                    for j, r in enumerate(head):
                        doc = table_doc(r, cols, title_col, f"{member} #{j}") if j >= start and cols else None
                        if doc:
                            yield {**doc, "source": self.name, "pos": [f, j + 1]}


# -----------------------------------------------------------------------------

def source_file_name(name):
    """'hf:owner/name' -> 'hf_owner_name': safe as a cache file name and a source label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def available_sources():
    return ["wikipedia", "simplewiki", "stackexchange", "gutenberg", *hf_presets(), "hf:<owner/name>",
            "kaggle:<owner/name>"]


def make_sources(names, topics=(), se_sites=None, delay=0.2, se_key=None, download_dir="."):
    available = {
        "wikipedia": lambda: Wikipedia("wikipedia", "en.wikipedia.org", 2000, topics, delay),
        "simplewiki": lambda: Wikipedia("simplewiki", "simple.wikipedia.org", 800, topics, delay),
        "stackexchange": lambda: StackExchange(**({"sites": se_sites} if se_sites else {}), key=se_key),
        "gutenberg": lambda: Gutenberg(),
        **hf_presets(),
    }
    out = []
    for n in names:
        if n.startswith("hf:"):
            out.append(hf_dataset(n[3:]))
        elif n.startswith("kaggle:"):
            out.append(Kaggle(n[7:], download_dir))
        elif n in available:
            out.append(available[n]())
        else:
            raise SystemExit(f"unknown source: {n}. Choose from: {', '.join(available_sources())}")
    return out
