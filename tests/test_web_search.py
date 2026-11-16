"""Explicit /search. Network is mocked. No live model and no real MemPalace."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tinytalk
import web_search
from provider import ProviderError


CLOCK = datetime(2026, 11, 16, 0, 18, tzinfo=timezone.utc)
RETRIEVED = "2026-11-16T00:18:00Z"
SEARCH_ENV = {"SEARXNG_URL": "http://searx.test"}
USAGE = "Use /search followed by what you want to look up on the web."


class ScriptedProvider(object):
    name = "scripted"
    label = "Script"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeSpeech(object):
    def speak_last(self, messages):
        return None

    def stop(self):
        return None


class MemorySpy(object):
    """Records every call. A web search must not write, and must not semantic-search."""

    palace = object()
    kg = None

    def __init__(self):
        self.events = []

    def _record(self, name, args, kwargs):
        self.events.append((name, args, kwargs))

    def search_facts(self, *args, **kwargs):
        self._record("search_facts", args, kwargs)
        return []

    def search_conversations(self, *args, **kwargs):
        self._record("search_conversations", args, kwargs)
        return []

    def save_exchange(self, *args, **kwargs):
        self._record("save_exchange", args, kwargs)

    def add_drawer(self, *args, **kwargs):
        self._record("add_drawer", args, kwargs)

    def update_drawer(self, *args, **kwargs):
        self._record("update_drawer", args, kwargs)

    def list_drawers(self, *args, **kwargs):
        self._record("list_drawers", args, kwargs)
        return {"drawers": [], "total": 0}

    def get_drawer(self, *args, **kwargs):
        self._record("get_drawer", args, kwargs)
        return {}


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


def _run_search(handler, query="cricket", env=None, clock=CLOCK, debug_log=None):
    if env is None:
        env = SEARCH_ENV
    client = _client(handler)
    try:
        return web_search.search(
            query, env=env, client=client, clock=clock, debug_log=debug_log
        )
    finally:
        client.close()


def _url(request):
    return str(request.url).split("?", 1)[0]


class RetrievalTests(unittest.TestCase):
    def test_missing_config_does_not_call_the_network(self):
        def handler(request):
            raise AssertionError("requested %s" % request.url)

        shown = io.StringIO()
        with redirect_stdout(shown):
            with patch.dict(os.environ, {"SEARXNG_URL": "http://ignored.example"}):
                with self.assertRaises(web_search.WebSearchError) as caught:
                    _run_search(handler, env={})
        self.assertEqual(str(caught.exception), web_search.NOT_CONFIGURED_MESSAGE)
        self.assertIsNone(caught.exception.detail)
        self.assertIn("SEARXNG_URL", str(caught.exception))
        self.assertIn("docker run -d -p 8888:8080 searxng/searxng", str(caught.exception))
        self.assertIn("search.formats: [html, json]", str(caught.exception))
        self.assertEqual(shown.getvalue(), "")
        self.assertFalse(web_search.configured(env={}))

    def test_numbering_dedupe_excerpt_cap_and_dropped_tags(self):
        seen = []
        page = (
            "<p>VISIBLE</p><script>SECRET</script><style>STYLETEXT</style>"
            "<nav>NAVTEXT</nav><footer>FOOTTEXT</footer><p>%s</p>"
        ) % ("a" * 4500)

        def handler(request):
            seen.append(_url(request))
            if request.url.path == "/search":
                self.assertEqual(request.url.params["format"], "json")
                self.assertEqual(request.url.params["q"], "cricket")
                return httpx.Response(200, json={"results": [
                    {"url": "https://example.com/1", "title": "One", "content": "snip one"},
                    {"url": "https://example.com/1", "title": "Duplicate", "content": "nope"},
                    {"url": "https://example.com/2", "title": "Two\nlines", "content": "snip two"},
                    {"url": "https://example.com/3", "title": "Three", "content": "snip three"},
                    {"url": "https://example.com/4", "title": "Four", "content": "snip four"},
                    {"url": "https://example.com/5", "title": "Five", "content": "fifth snippet"},
                    {"url": "https://example.com/6", "title": "Six", "content": "sixth"},
                ]})
            if _url(request) == "https://example.com/1":
                return httpx.Response(200, text=page, headers={"content-type": "text/html"})
            if _url(request) == "https://example.com/5":
                return httpx.Response(500, text="down")
            if _url(request) == "https://example.com/6":
                raise AssertionError("sixth result was fetched")
            return httpx.Response(200, text="<p>%s page</p>" % request.url.path.strip("/"))

        found = _run_search(handler)
        self.assertEqual([source.number for source in found], [1, 2, 3, 4, 5])
        self.assertEqual(
            [source.url for source in found],
            [
                "https://example.com/1",
                "https://example.com/2",
                "https://example.com/3",
                "https://example.com/4",
                "https://example.com/5",
            ],
        )
        self.assertEqual(found[0].title, "One")
        self.assertNotIn("Duplicate", [source.title for source in found])
        self.assertEqual(found[1].title, "Two lines")
        self.assertEqual(len(found[0].excerpt), 4000)
        self.assertTrue(found[0].excerpt.startswith("VISIBLE "))
        self.assertNotIn("SECRET", found[0].excerpt)
        self.assertNotIn("STYLETEXT", found[0].excerpt)
        self.assertNotIn("NAVTEXT", found[0].excerpt)
        self.assertNotIn("FOOTTEXT", found[0].excerpt)
        self.assertFalse(found[0].snippet_only)
        self.assertNotIn("snip one", found[0].excerpt)
        self.assertTrue(found[4].snippet_only)
        self.assertEqual(found[4].excerpt, "fifth snippet")
        self.assertEqual(found[0].retrieved, RETRIEVED)
        self.assertEqual(found[4].retrieved, RETRIEVED)
        self.assertNotIn("https://example.com/6", seen)

    def test_oversize_non_http_and_redirects(self):
        requested = []

        def handler(request):
            requested.append(_url(request))
            if request.url.path == "/search":
                return httpx.Response(200, json={"results": [
                    {
                        "url": "http://example.com/start",
                        "title": "Hop",
                        "content": "start snippet",
                    },
                    {
                        "url": "http://example.com/loop",
                        "title": "Loop",
                        "content": "loop snippet",
                    },
                    {
                        "url": "ftp://files.example/secret",
                        "title": "File",
                        "content": "ftp snippet",
                    },
                    {
                        "url": "https://example.com/huge",
                        "title": "Huge",
                        "content": "huge snippet",
                    },
                ]})
            if _url(request) == "http://example.com/start":
                return httpx.Response(302, headers={"location": "/land"})
            if _url(request) == "http://example.com/land":
                return httpx.Response(200, text="<p>landed ok</p>")
            if _url(request).startswith("http://example.com/loop"):
                hop = _url(request).rsplit("/", 1)[-1]
                if hop == "loop":
                    nxt = "http://example.com/loop/1"
                else:
                    nxt = "http://example.com/loop/%s" % (int(hop) + 1)
                return httpx.Response(302, headers={"location": nxt})
            if _url(request) == "https://example.com/huge":
                return httpx.Response(
                    200,
                    content=b"h" * (web_search.MAX_BYTES + 1),
                    headers={"content-type": "text/html"},
                )
            if request.url.scheme == "ftp":
                raise AssertionError("ftp was fetched")
            raise AssertionError("unexpected %s" % request.url)

        found = _run_search(handler)
        by_url = {source.url: source for source in found}
        self.assertEqual(by_url["http://example.com/start"].excerpt, "landed ok")
        self.assertFalse(by_url["http://example.com/start"].snippet_only)
        self.assertTrue(by_url["http://example.com/loop"].snippet_only)
        self.assertEqual(by_url["http://example.com/loop"].excerpt, "loop snippet")
        self.assertTrue(by_url["ftp://files.example/secret"].snippet_only)
        self.assertEqual(by_url["ftp://files.example/secret"].excerpt, "ftp snippet")
        self.assertTrue(by_url["https://example.com/huge"].snippet_only)
        self.assertEqual(by_url["https://example.com/huge"].excerpt, "huge snippet")
        self.assertNotIn("ftp://files.example/secret", requested)
        self.assertEqual(
            [url for url in requested if "/loop" in url],
            [
                "http://example.com/loop",
                "http://example.com/loop/1",
                "http://example.com/loop/2",
                "http://example.com/loop/3",
            ],
        )
        self.assertNotIn("http://example.com/loop/4", requested)

    def test_no_results_and_dead_pages_are_plain_errors(self):
        def empty(request):
            return httpx.Response(200, json={"results": []})

        with self.assertRaises(web_search.WebSearchError) as caught:
            _run_search(empty)
        self.assertEqual(str(caught.exception), web_search.NO_RESULTS_MESSAGE)
        self.assertNotIn("http", str(caught.exception).lower())

        def dead(request):
            if request.url.path == "/search":
                return httpx.Response(200, json={"results": [
                    {"url": "https://example.com/a", "title": "A"},
                    {"url": "https://example.com/b", "title": "B", "content": "   "},
                ]})
            return httpx.Response(500, text="nope")

        shown = io.StringIO()
        with redirect_stdout(shown):
            with self.assertRaises(web_search.WebSearchError) as caught:
                _run_search(dead)
        self.assertEqual(str(caught.exception), web_search.UNREADABLE_MESSAGE)
        self.assertNotIn("HTTP", str(caught.exception))
        self.assertIn("HTTP 500", caught.exception.detail)
        self.assertEqual(shown.getvalue(), "")

    def test_a_down_backend_is_one_plain_error(self):
        def handler(request):
            raise httpx.ConnectError("connection refused")

        shown = io.StringIO()
        with redirect_stdout(shown):
            with self.assertRaises(web_search.WebSearchError) as caught:
                _run_search(handler)
        self.assertEqual(str(caught.exception), web_search.UNREACHABLE_MESSAGE)
        self.assertIn("ConnectError", caught.exception.detail)
        self.assertNotIn("ConnectError", str(caught.exception))
        self.assertNotIn("connection refused", str(caught.exception))
        self.assertEqual(shown.getvalue(), "")


class SearchCommandTests(unittest.TestCase):
    def _line(self, prompt, provider, memory, sources, messages=None, debug=False, handler=None):
        if messages is None:
            messages = []
        shown = io.StringIO()
        env = {"TINYTALK_DEBUG": ""}
        if handler is None:
            env["SEARXNG_URL"] = ""
        else:
            env["SEARXNG_URL"] = "http://searx.test"
        patches = [patch.dict("os.environ", env, clear=False)]
        if handler is not None:
            patches.append(patch("web_search._new_client", return_value=_client(handler)))
        for item in patches:
            item.start()
        try:
            with redirect_stdout(shown):
                updated = tinytalk.handle_user_line(
                    prompt,
                    messages,
                    provider,
                    memory,
                    "SOUL",
                    FakeSpeech(),
                    sources,
                    debug=debug,
                )
        finally:
            for item in reversed(patches):
                item.stop()
        return updated, shown.getvalue()

    def test_no_query_prints_usage_and_does_not_search(self):
        provider = ScriptedProvider([])
        sources = tinytalk.SessionSources()
        sources.replace([{"kind": "saved fact", "text": "keep me", "id": "keep"}])
        history = [{"role": "assistant", "content": "earlier"}]

        def handler(request):
            raise AssertionError("searched")

        for prompt in ("/search", "/search   ", "/searchfoo", "/Search"):
            updated, shown = self._line(prompt, provider, None, sources, list(history), handler=handler)
            self.assertEqual(updated, history)
            self.assertEqual(sources.records, [{"kind": "saved fact", "text": "keep me", "id": "keep"}])
            self.assertIn(USAGE, shown)
            self.assertNotIn("WEB SOURCES", shown)
        self.assertEqual(provider.calls, [])

    def test_missing_config_prints_how_to_set_it_and_stops(self):
        provider = ScriptedProvider([AssertionError("model")])
        sources = tinytalk.SessionSources()
        kept = [{"kind": "saved fact", "text": "keep me", "id": "keep"}]
        sources.replace(list(kept))
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]

        def handler(request):
            raise AssertionError("searched")

        updated, shown = self._line(
            "/search live cricket score", provider, MemorySpy(), sources, list(history)
        )
        self.assertEqual(updated, history)
        self.assertEqual(sources.records, kept)
        self.assertEqual(provider.calls, [])
        self.assertIn("Web search is not configured.", shown)
        self.assertIn("Set SEARXNG_URL", shown)
        self.assertIn("docker run -d -p 8888:8080 searxng/searxng", shown)
        self.assertIn("search.formats: [html, json]", shown)
        self.assertNotIn("Script:", shown)

    def test_prompt_contains_numbered_sources_and_the_citation_rule(self):
        provider = ScriptedProvider(["Alpha is first [1]. Beta is second [2]."])
        sources = tinytalk.SessionSources()

        def handler(request):
            if request.url.path == "/search":
                return httpx.Response(200, json={"results": [
                    {
                        "url": "https://example.com/alpha",
                        "title": "Alpha",
                        "content": "snippet alpha",
                    },
                    {
                        "url": "https://example.com/beta",
                        "title": "Beta",
                        "content": "snippet beta",
                    },
                ]})
            name = request.url.path.strip("/")
            return httpx.Response(200, text="<p>%s page body</p>" % name)

        updated, shown = self._line(
            "/search Lisbon bakeries", provider, None, sources, handler=handler
        )
        self.assertEqual(len(provider.calls), 1)
        prompt = "\n".join(message["content"] for message in provider.calls[0])
        self.assertIn("WEB SOURCES (retrieved ", prompt)
        self.assertIn("— live web results, not saved memories)", prompt)
        self.assertIn("[1] Alpha — https://example.com/alpha", prompt)
        self.assertIn("alpha page body", prompt)
        self.assertIn("[2] Beta — https://example.com/beta", prompt)
        self.assertIn("beta page body", prompt)
        self.assertIn("Cite every web-derived claim with its [n].", prompt)
        self.assertIn("never invent one", prompt)
        self.assertIn("silently filling from training", prompt)
        self.assertEqual(provider.calls[0][-1]["content"], "/search Lisbon bakeries")
        for message in provider.calls[0]:
            if message["role"] == "system":
                self.assertTrue((message["content"] or "").strip())
        self.assertEqual(updated[-1]["content"], "Alpha is first [1]. Beta is second [2].")
        self.assertEqual([record["kind"] for record in sources.records], ["web source", "web source"])
        self.assertEqual(sources.records[0]["selection"], "explicit web search")
        self.assertEqual(sources.records[0]["url"], "https://example.com/alpha")
        self.assertEqual(sources.records[0]["number"], 1)
        self.assertRegex(sources.records[0]["retrieved_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        self.assertIn(sources.records[0]["retrieved_at"], prompt)
        report = tinytalk.format_sources_report(sources.records)
        self.assertIn("These web sources were included", report)
        self.assertIn("not saved memories", report)
        self.assertIn("selection: explicit web search", report)
        self.assertIn("url: https://example.com/alpha", report)
        self.assertIn("This is when the page was retrieved, not when a fact changed.", report)
        self.assertNotIn("These saved memories were included", report)
        self.assertIn("Script: Alpha is first [1]. Beta is second [2].", shown)

    def test_backend_down_leaves_sources_and_history_untouched(self):
        provider = ScriptedProvider([AssertionError("model")])
        sources = tinytalk.SessionSources()
        kept = [{"kind": "saved fact", "text": "keep me", "id": "keep"}]
        sources.replace(list(kept))
        history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "hi"}]

        def handler(request):
            raise httpx.ConnectError("connection refused")

        updated, shown = self._line(
            "/search cricket", provider, MemorySpy(), sources, list(history), handler=handler
        )
        self.assertEqual(updated, history)
        self.assertEqual(sources.records, kept)
        self.assertEqual(provider.calls, [])
        self.assertIn(web_search.UNREACHABLE_MESSAGE, shown)
        self.assertNotIn("ConnectError", shown)
        self.assertNotIn("connection refused", shown)
        self.assertNotIn("Script:", shown)

        debug_shown = io.StringIO()
        with redirect_stdout(debug_shown):
            with patch.dict("os.environ", {"SEARXNG_URL": "http://searx.test", "TINYTALK_DEBUG": ""}, clear=False):
                with patch("web_search._new_client", return_value=_client(handler)):
                    tinytalk.handle_user_line(
                        "/search cricket",
                        list(history),
                        provider,
                        MemorySpy(),
                        "SOUL",
                        FakeSpeech(),
                        sources,
                        debug=True,
                    )
        self.assertIn(web_search.UNREACHABLE_MESSAGE, debug_shown.getvalue())
        self.assertIn("ConnectError", debug_shown.getvalue())
        self.assertEqual(sources.records, kept)

    def test_no_results_and_dead_pages_leave_sources_untouched(self):
        provider = ScriptedProvider([AssertionError("model")])
        history = [{"role": "assistant", "content": "stay"}]
        kept = [{"kind": "saved fact", "text": "keep me"}]

        def empty(request):
            return httpx.Response(200, json={"results": []})

        def dead(request):
            if request.url.path == "/search":
                return httpx.Response(200, json={"results": [
                    {"url": "https://example.com/a", "title": "A"},
                ]})
            return httpx.Response(500, text="nope")

        for handler, message in (
            (empty, web_search.NO_RESULTS_MESSAGE),
            (dead, web_search.UNREADABLE_MESSAGE),
        ):
            sources = tinytalk.SessionSources()
            sources.replace(list(kept))
            updated, shown = self._line(
                "/search cricket", provider, None, sources, list(history), handler=handler
            )
            self.assertEqual(updated, history)
            self.assertEqual(sources.records, kept)
            self.assertIn(message, shown)
            self.assertNotIn("HTTP", shown)
        self.assertEqual(provider.calls, [])

    def test_memory_spy_shows_no_mempalace_writes(self):
        provider = ScriptedProvider(["Plankton drifts [1]."])
        sources = tinytalk.SessionSources()
        memory = MemorySpy()

        def handler(request):
            if request.url.path == "/search":
                return httpx.Response(200, json={"results": [
                    {
                        "url": "https://example.com/plankton",
                        "title": "Plankton",
                        "content": "do not store this snippet",
                    },
                ]})
            return httpx.Response(
                200,
                text="<p>Plankton drifts in the ocean and is not a saved memory.</p>",
            )

        updated, shown = self._line(
            "/search Remember this: plankton lives in the sea",
            provider,
            memory,
            sources,
            handler=handler,
        )
        self.assertEqual(updated[-1]["content"], "Plankton drifts [1].")
        prompt = provider.calls[0]
        body = "\n".join(message["content"] for message in prompt)
        self.assertIn("Plankton drifts in the ocean", body)
        self.assertIn("[1] Plankton — https://example.com/plankton", body)
        self.assertIn("The saved preferred_name is unavailable.", body)
        names = [event[0] for event in memory.events]
        self.assertNotIn("search_facts", names)
        self.assertNotIn("search_conversations", names)
        self.assertNotIn("save_exchange", names)
        self.assertNotIn("add_drawer", names)
        self.assertNotIn("update_drawer", names)
        blob = repr(memory.events)
        self.assertNotIn("example.com/plankton", blob)
        self.assertNotIn("Plankton drifts", blob)
        self.assertNotIn("do not store this snippet", blob)
        web = [record for record in sources.records if record["kind"] == "web source"]
        self.assertEqual(len(web), 1)
        self.assertEqual(web[0]["selection"], "explicit web search")
        report = tinytalk.format_sources_report(sources.records)
        memory_part, web_part = report.split("These web sources were included", 1)
        self.assertIn("These saved memories were included", memory_part)
        self.assertNotIn("example.com/plankton", memory_part)
        self.assertIn("url: https://example.com/plankton", web_part)
        self.assertIn("explicit web search", web_part)
        self.assertIn("not when a fact changed", web_part)
        again, listed = self._line("/sources", provider, memory, sources, updated)
        self.assertEqual(again, updated)
        self.assertIn("web source", listed)
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("Script: Plankton drifts [1].", shown)

    def test_an_ordinary_turn_does_not_search(self):
        provider = ScriptedProvider(["Just hello."])
        sources = tinytalk.SessionSources()

        def handler(request):
            raise AssertionError("searched %s" % request.url)

        updated, shown = self._line(
            "What is the weather in Lisbon?",
            provider,
            None,
            sources,
            handler=handler,
        )
        self.assertEqual(len(provider.calls), 1)
        sent = "\n".join(message["content"] for message in provider.calls[0])
        self.assertNotIn("WEB SOURCES", sent)
        self.assertEqual(updated[-1]["content"], "Just hello.")
        self.assertEqual(sources.records, [])
        self.assertIn("Script: Just hello.", shown)

    def test_a_failed_model_call_does_not_replace_sources_or_save(self):
        provider = ScriptedProvider([ProviderError("the model failed")])
        sources = tinytalk.SessionSources()
        kept = [{"kind": "saved fact", "text": "keep me", "id": "keep"}]
        sources.replace(list(kept))
        memory = MemorySpy()

        def handler(request):
            if request.url.path == "/search":
                return httpx.Response(200, json={"results": [
                    {"url": "https://example.com/a", "title": "A", "content": "snip"},
                ]})
            return httpx.Response(200, text="<p>page</p>")

        updated, shown = self._line(
            "/search cricket", provider, memory, sources, handler=handler
        )
        self.assertEqual(updated, [])
        self.assertEqual(sources.records, kept)
        self.assertIn("Nothing was saved.", shown)
        self.assertEqual([event[0] for event in memory.events], [])

    def test_memory_only_sources_report_stays_the_same(self):
        report = tinytalk.format_sources_report([{
            "kind": "saved fact",
            "text": "my test spaceship is named Enterprise",
            "id": "drawer_enterprise",
            "filed_at": "2026-09-27T23:08:05",
        }])
        self.assertEqual(
            report,
            "\n".join([
                "These saved memories were included in the context for the last "
                "successful answer. TinyTalk does not claim the model used every one.",
                "1. saved fact",
                "id: drawer_enterprise",
                "filed: 2026-09-27T23:08:05",
                "This is when the record was filed, not necessarily when the fact changed.",
                "my test spaceship is named Enterprise",
            ]),
        )
        empty = tinytalk.format_sources_report([])
        self.assertIn("No saved memories were included", empty)
        self.assertIn("does not mean no memories exist", empty)


if __name__ == "__main__":
    unittest.main()
