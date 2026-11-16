"""Explicit live web search. Retrieval only.

This module never calls the model and never reads or writes memory.
SearxNG is the only backend. SEARXNG_URL has no default.
"""

import json
import os
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx


MAX_RESULTS = 5
EXCERPT_CHARS = 4000
MAX_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 3
CONNECT_SECONDS = 10.0
TOTAL_SECONDS = 20.0
USER_AGENT = "TinyTalk"
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
_SKIP_TAGS = {"script", "style", "nav", "footer"}

NOT_CONFIGURED_MESSAGE = (
    "Web search is not configured.\n"
    "Set SEARXNG_URL to your own SearxNG instance. There is no default.\n"
    "Example: SEARXNG_URL=http://127.0.0.1:8888\n"
    "Enable JSON in that instance's settings.yml with search.formats: [html, json].\n"
    "docker run -d -p 8888:8080 searxng/searxng"
)
UNREACHABLE_MESSAGE = "Web search could not reach the search service."
NO_RESULTS_MESSAGE = "Web search found no results."
UNREADABLE_MESSAGE = "Web search could not read any result page."


class WebSearchError(Exception):
    """One plain search failure. Detail is for TINYTALK_DEBUG, not the user."""

    def __init__(self, message, detail=None):
        super(WebSearchError, self).__init__(message)
        self.detail = detail


class WebSource(object):
    """One retrieved page. number is 1-based among the records actually returned."""

    def __init__(self, number, title, url, excerpt, retrieved, snippet_only):
        self.number = number
        self.title = title
        self.url = url
        self.excerpt = excerpt
        self.retrieved = retrieved
        self.snippet_only = bool(snippet_only)


class _TextExtractor(HTMLParser):
    """Visible text, without script, style, nav, or footer."""

    def __init__(self):
        super(_TextExtractor, self).__init__(convert_charrefs=True)
        self._skip = 0
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if (tag or "").lower() in _SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if (tag or "").lower() in _SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip or not data:
            return
        if self._parts and not self._parts[-1][-1].isspace() and not data[0].isspace():
            self._parts.append(" ")
        self._parts.append(data)

    def text(self):
        return " ".join("".join(self._parts).split())


def configured(env=None):
    """True when SEARXNG_URL is set to a non-empty value."""
    return bool(_base_url(env))


def search(query, env=None, client=None, clock=None, debug_log=None):
    """Return up to five WebSource records for one explicit query.

    Raises WebSearchError with a plain message. Page failures keep the
    SearxNG snippet and set snippet_only. Debug lines are not printed here;
    pass debug_log to receive them, and read WebSearchError.detail otherwise.
    """
    base = _base_url(env)
    if not base:
        raise WebSearchError(NOT_CONFIGURED_MESSAGE)
    owns_client = client is None
    if owns_client:
        client = _new_client()
    try:
        return _search(query, base, client, clock, debug_log)
    finally:
        if owns_client:
            client.close()


def fetch(url, client=None):
    """GET one http(s) page and return extracted text, capped at 4000 characters.

    Raises WebSearchError when the page cannot be read. A caller that still
    has a search snippet can keep that snippet instead.
    """
    owns_client = client is None
    if owns_client:
        client = _new_client()
    try:
        _content_type, body = _get_bytes(client, url)
    finally:
        if owns_client:
            client.close()
    text = extract_text(_decode_body(_content_type, body))
    if len(text) > EXCERPT_CHARS:
        return text[:EXCERPT_CHARS]
    return text


def extract_text(html):
    """Drop script, style, nav, and footer, then collapse whitespace."""
    parser = _TextExtractor()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception:
        pass
    return parser.text()


def _search(query, base, client, clock, debug_log):
    try:
        content_type, body = _get_bytes(
            client,
            base + "/search",
            params={"q": query, "format": "json"},
        )
    except WebSearchError as exc:
        raise WebSearchError(UNREACHABLE_MESSAGE, detail=exc.detail)
    del content_type
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise WebSearchError(UNREACHABLE_MESSAGE, detail="unreadable JSON: %s" % exc)
    if not isinstance(payload, dict):
        raise WebSearchError(UNREACHABLE_MESSAGE, detail="SearxNG JSON was not an object")
    results = payload.get("results")
    if not isinstance(results, list):
        results = []
    chosen = _top_results(results)
    if not chosen:
        raise WebSearchError(NO_RESULTS_MESSAGE)

    retrieved = _retrieved_at(clock)
    kept = []
    notes = []
    for item in chosen:
        url = _result_url(item)
        title = _result_title(item, url)
        snippet = _snippet(item)
        failure = None
        try:
            excerpt = fetch(url, client=client)
        except WebSearchError as exc:
            excerpt = ""
            failure = exc.detail or str(exc)
        if excerpt.strip():
            kept.append(WebSource(0, title, url, excerpt, retrieved, False))
            continue
        note = failure or ("empty page %s" % url)
        if snippet:
            notes.append(note)
            kept.append(
                WebSource(0, title, url, snippet[:EXCERPT_CHARS], retrieved, True)
            )
            continue
        notes.append(note)
    if not kept:
        detail = "; ".join(note for note in notes if note) or "no page could be read"
        raise WebSearchError(UNREADABLE_MESSAGE, detail=detail)
    for number, source in enumerate(kept, 1):
        source.number = number
    if debug_log is not None:
        for note in notes:
            if note:
                debug_log(note)
    return kept


def _top_results(results):
    """First five results with a URL, skipping a URL that already appeared."""
    chosen = []
    seen = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        url = _result_url(item)
        if not url or url in seen:
            continue
        seen.add(url)
        chosen.append(item)
        if len(chosen) >= MAX_RESULTS:
            break
    return chosen


def _result_url(item):
    url = item.get("url")
    if not isinstance(url, str):
        return ""
    return url.strip()


def _result_title(item, url):
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        return url
    return " ".join(title.split())


def _snippet(item):
    raw = item.get("content")
    if not isinstance(raw, str) or not raw.strip():
        raw = item.get("snippet")
    if not isinstance(raw, str):
        return ""
    return " ".join(raw.split())


def _retrieved_at(clock):
    if clock is None:
        moment = datetime.now(timezone.utc)
    elif callable(clock):
        moment = clock()
    else:
        moment = clock
    if isinstance(moment, datetime):
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(moment)


def _new_client():
    return httpx.Client(
        timeout=httpx.Timeout(TOTAL_SECONDS, connect=CONNECT_SECONDS),
        follow_redirects=False,
    )


def _allowed_url(url):
    if not isinstance(url, str):
        return False
    parsed = urlparse(url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _get_bytes(client, url, params=None):
    """GET one URL. At most 3 redirects, 10s connect, 20s total, 2 MB."""
    current = url
    send_params = params
    started = time.monotonic()
    redirects = 0
    while True:
        if not _allowed_url(current):
            raise WebSearchError(
                UNREADABLE_MESSAGE,
                detail="refused non-http URL %s" % current,
            )
        remaining = TOTAL_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            raise WebSearchError(
                UNREADABLE_MESSAGE,
                detail="timed out fetching %s" % current,
            )
        timeout = httpx.Timeout(remaining, connect=min(CONNECT_SECONDS, remaining))
        response = None
        try:
            request = client.build_request(
                "GET",
                current,
                params=send_params,
                timeout=timeout,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html, application/json;q=0.9, text/plain;q=0.8, */*;q=0.1",
                },
            )
            response = client.send(request, stream=True, follow_redirects=False)
            status = response.status_code
            if status in REDIRECT_STATUSES:
                location = response.headers.get("location")
                redirects += 1
                if redirects > MAX_REDIRECTS or not location:
                    raise WebSearchError(
                        UNREADABLE_MESSAGE,
                        detail="stopped after %s redirects at %s" % (redirects, current),
                    )
                current = urljoin(current, location.strip())
                send_params = None
                continue
            if status != 200:
                raise WebSearchError(
                    UNREADABLE_MESSAGE,
                    detail="HTTP %s for %s" % (status, current),
                )
            content_type = response.headers.get("content-type")
            body = _read_capped(response)
            return content_type, body
        except WebSearchError:
            raise
        except Exception as exc:
            raise WebSearchError(
                UNREADABLE_MESSAGE,
                detail="%s: %s" % (type(exc).__name__, exc),
            )
        finally:
            if response is not None:
                response.close()


def _read_capped(response):
    declared = response.headers.get("content-length")
    if declared:
        try:
            if int(declared) > MAX_BYTES:
                raise WebSearchError(
                    UNREADABLE_MESSAGE,
                    detail="response larger than 2 MB",
                )
        except ValueError:
            pass
    chunks = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > MAX_BYTES:
            raise WebSearchError(
                UNREADABLE_MESSAGE,
                detail="response larger than 2 MB",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _decode_body(content_type, body):
    encoding = "utf-8"
    if content_type:
        for part in content_type.split(";"):
            item = part.strip()
            if item.lower().startswith("charset="):
                encoding = item.split("=", 1)[1].strip().strip('"').strip("'") or "utf-8"
                break
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _base_url(env):
    source = os.environ if env is None else env
    return (source.get("SEARXNG_URL") or "").strip().rstrip("/")
