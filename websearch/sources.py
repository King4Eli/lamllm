"""Text sources LamLLM can learn from.

Each source's documents() is a generator yielding dicts:
    {"source": ..., "title": ..., "text": ..., "qa": [[question, answer], ...]}
Text is cleaned to plain ASCII. "qa" holds question/answer pairs the model is
also trained on, so it learns to respond to "Q: ...\\nA:" prompts.

Requests go one at a time per source, with a delay, and back off when a site
says it is overloaded.
"""

import gzip
import html
import json
import re
import time
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

    def documents(self):
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


class StackExchange:
    API = "https://api.stackexchange.com/2.3"

    def __init__(self, sites=("history", "biology", "physics", "astronomy", "cooking", "english", "travel", "diy"),
                 delay=1.0):
        self.name = "stackexchange"
        self.sites = list(sites)
        self.delay = delay

    def _get(self, path, **params):
        time.sleep(self.delay)
        data = get_json(f"{self.API}/{path}", params)
        if data.get("backoff"):  # the API asks clients to pause when told to
            time.sleep(data["backoff"])
        return data

    def documents(self):
        page = 1
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
                           "qa": [[title, short]] if len(short) > 20 else []}
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

    def documents(self):
        url = f"{self.CATALOG}?languages=en"
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
                yield {"source": self.name, "title": title, "text": text, "qa": []}
            url = data.get("next")


def make_sources(names, topics=(), se_sites=None, delay=0.2):
    available = {
        "wikipedia": lambda: Wikipedia("wikipedia", "en.wikipedia.org", 2000, topics, delay),
        "simplewiki": lambda: Wikipedia("simplewiki", "simple.wikipedia.org", 800, topics, delay),
        "stackexchange": lambda: StackExchange(se_sites) if se_sites else StackExchange(),
        "gutenberg": lambda: Gutenberg(),
    }
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"unknown source(s): {', '.join(unknown)}. Choose from: {', '.join(available)}")
    return [available[n]() for n in names]
