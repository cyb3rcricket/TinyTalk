import io
import json
import os
import sys
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import APIConnectionError, AuthenticationError, NotFoundError, RateLimitError

import tinytalk
from speech import SpeechController
from provider import (
    DEFAULT_OLLAMA_MODEL,
    DOCUMENTED_XAI_MODEL,
    XAI_BASE_URL,
    ConfigError,
    GrokProvider,
    OllamaProvider,
    ProviderError,
    build_provider,
    load_settings,
    text_from_chat_completion,
    text_from_responses,
)


def status_error(cls, status, message, url):
    request = httpx.Request("POST", url)
    response = httpx.Response(status, request=request)
    return cls(message, response=response, body={"error": message})


class FakeCompletions(object):
    def __init__(self, results, calls):
        self.results = results
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeResponses(object):
    def __init__(self, results, calls):
        self.results = results
        self.calls = calls

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeChat(object):
    def __init__(self, completions):
        self.completions = completions


class FakeClient(object):
    def __init__(self, chat_results=None, response_results=None):
        self.chat_calls = []
        self.response_calls = []
        self.chat = FakeChat(FakeCompletions(chat_results or [], self.chat_calls))
        self.responses = FakeResponses(response_results or [], self.response_calls)


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


class StopPlayer(object):
    def __init__(self):
        self.stop_calls = 0

    def play(self, wav_bytes, still_current):
        return False

    def stop(self):
        self.stop_calls += 1


class RecordingMemory(object):
    def __init__(self, facts=None, conversations=None):
        self.palace = object()
        self.kg = None
        self.facts = list(facts or [])
        self.conversations = list(conversations or [])
        self.saved = []

    def list_facts(self):
        return list(self.facts)

    def search_facts(self, query):
        return list(self.facts)

    def search_conversations(self, query):
        return list(self.conversations)

    def save_exchange(self, user_prompt, response):
        self.saved.append((user_prompt, response))


class SettingsTests(unittest.TestCase):
    def test_ollama_is_the_default_model_and_ignores_a_missing_xai_key(self):
        settings = load_settings({})
        self.assertEqual(settings.provider, "ollama")
        self.assertEqual(settings.ollama_model, DEFAULT_OLLAMA_MODEL)
        self.assertEqual(settings.ollama_base_url, "http://localhost:11434/v1")
        self.assertEqual(settings.xai_api_key, "")

    def test_provider_name_is_case_insensitive(self):
        settings = load_settings({
            "TINYTALK_PROVIDER": " Grok ",
            "XAI_API_KEY": "test-key",
            "XAI_MODEL": DOCUMENTED_XAI_MODEL,
        })
        self.assertEqual(settings.provider, "grok")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ConfigError) as caught:
            load_settings({"TINYTALK_PROVIDER": "gemini"})
        self.assertIn("ollama", str(caught.exception))
        self.assertIn("grok", str(caught.exception))

    def test_grok_requires_key_and_model(self):
        with self.assertRaises(ConfigError) as missing_key:
            load_settings({"TINYTALK_PROVIDER": "grok", "XAI_MODEL": DOCUMENTED_XAI_MODEL})
        self.assertIn("XAI_API_KEY", str(missing_key.exception))
        self.assertIn("did not fall back", str(missing_key.exception))

        with self.assertRaises(ConfigError) as missing_model:
            load_settings({"TINYTALK_PROVIDER": "grok", "XAI_API_KEY": "test-key"})
        self.assertIn("XAI_MODEL", str(missing_model.exception))
        self.assertIn(DOCUMENTED_XAI_MODEL, str(missing_model.exception))

    def test_ollama_mode_does_not_create_a_grok_client(self):
        created = []

        def factory(**kwargs):
            created.append(kwargs)
            return FakeClient()

        settings = load_settings({
            "TINYTALK_PROVIDER": "ollama",
            "XAI_API_KEY": "super-secret-key",
            "XAI_MODEL": DOCUMENTED_XAI_MODEL,
        })
        provider = build_provider(settings, client_factory=factory)
        self.assertIsInstance(provider, OllamaProvider)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["base_url"], "http://localhost:11434/v1")
        self.assertEqual(created[0]["api_key"], "ollama")
        self.assertEqual(created[0]["max_retries"], 0)
        self.assertNotIn("super-secret-key", str(created[0]))
        self.assertNotEqual(created[0]["base_url"], XAI_BASE_URL)

    def test_missing_grok_settings_do_not_create_a_client(self):
        created = []
        settings = load_settings({})
        settings.provider = "grok"
        settings.xai_api_key = ""
        settings.xai_model = ""
        with self.assertRaises(ConfigError):
            build_provider(settings, client_factory=lambda **kwargs: created.append(kwargs))
        self.assertEqual(created, [])

    def test_grok_client_uses_the_responses_base_url(self):
        created = []

        def factory(**kwargs):
            created.append(kwargs)
            return FakeClient()

        settings = load_settings({
            "TINYTALK_PROVIDER": "grok",
            "XAI_API_KEY": "test-key",
            "XAI_MODEL": DOCUMENTED_XAI_MODEL,
        })
        provider = build_provider(settings, client_factory=factory)
        self.assertIsInstance(provider, GrokProvider)
        self.assertEqual(created[0]["base_url"], XAI_BASE_URL)
        self.assertEqual(created[0]["api_key"], "test-key")
        self.assertEqual(created[0]["max_retries"], 0)
        self.assertEqual(provider.model, DOCUMENTED_XAI_MODEL)


class NormalizationTests(unittest.TestCase):
    def test_chat_completion_text(self):
        payload = {"choices": [{"message": {"content": "  hello  "}}]}
        self.assertEqual(text_from_chat_completion(payload), "  hello  ")

    def test_responses_output_text_and_message_blocks(self):
        payload = {
            "status": "completed",
            "output_text": "",
            "output": [
                {"type": "reasoning", "summary": [{"text": "hidden"}]},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Enterprise"},
                    ],
                },
            ],
        }
        self.assertEqual(text_from_responses(payload), "Enterprise")

    def test_responses_prefers_output_text(self):
        payload = {
            "status": "completed",
            "output_text": "from output_text",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "other"}]}],
        }
        self.assertEqual(text_from_responses(payload), "from output_text")

    def test_incomplete_response_is_an_error(self):
        with self.assertRaises(ProviderError) as caught:
            text_from_responses({"status": "incomplete", "output_text": "partial"})
        self.assertIn("incomplete", str(caught.exception))
        self.assertIn("did not switch", str(caught.exception))


class ApiFailureTests(unittest.TestCase):
    def test_auth_failure_is_not_retried_or_saved(self):
        secret = "xai-live-looking-key"
        error = status_error(
            AuthenticationError,
            401,
            "bad key %s" % secret,
            "https://api.x.ai/v1/responses",
        )
        client = FakeClient(response_results=[error, error])
        provider = GrokProvider(client, DOCUMENTED_XAI_MODEL, api_key=secret, sleep=lambda _s: None)
        memory = RecordingMemory()
        with self.assertRaises(ProviderError) as caught:
            provider.complete([{"role": "user", "content": "hi"}])
        message = str(caught.exception)
        self.assertIn("401", message)
        self.assertIn("XAI_API_KEY", message)
        self.assertNotIn(secret, message)
        self.assertIn("did not switch", message)
        self.assertEqual(len(client.response_calls), 1)

        history = tinytalk.handle_turn("hi", [], provider, memory, soul="SOUL")
        self.assertEqual(history, [])
        self.assertEqual(memory.saved, [])

    def test_rate_limit_retries_then_succeeds(self):
        error = status_error(RateLimitError, 429, "slow down", "https://api.x.ai/v1/responses")
        client = FakeClient(response_results=[
            error,
            {"status": "completed", "output_text": "ok"},
        ])
        delays = []
        provider = GrokProvider(
            client,
            DOCUMENTED_XAI_MODEL,
            api_key="k",
            sleep=delays.append,
        )
        text = provider.complete([{"role": "user", "content": "hi"}])
        self.assertEqual(text, "ok")
        self.assertEqual(delays, [0.5])
        self.assertEqual(len(client.response_calls), 2)

    def test_rate_limit_is_explained_after_the_last_attempt(self):
        error = status_error(RateLimitError, 429, "slow down", "https://api.x.ai/v1/responses")
        client = FakeClient(response_results=[error, error, error])
        provider = GrokProvider(client, "grok-4.7", api_key="k", sleep=lambda _s: None)
        with self.assertRaises(ProviderError) as caught:
            provider.complete([{"role": "user", "content": "hi"}])
        self.assertIn("429", str(caught.exception))
        self.assertEqual(len(client.response_calls), 3)

    def test_connection_error_names_the_service(self):
        request = httpx.Request("POST", "http://localhost:11434/v1/chat/completions")
        client = FakeClient(chat_results=[APIConnectionError(request=request)])
        provider = OllamaProvider(client, DEFAULT_OLLAMA_MODEL, sleep=lambda _s: None)
        with self.assertRaises(ProviderError) as caught:
            provider.complete([{"role": "user", "content": "hi"}])
        self.assertIn("Ollama", str(caught.exception))
        self.assertIn("did not switch", str(caught.exception))

    def test_missing_model_names_the_configured_id(self):
        error = status_error(NotFoundError, 404, "no such model", "https://api.x.ai/v1/responses")
        client = FakeClient(response_results=[error])
        provider = GrokProvider(client, "not-a-model", api_key="k", sleep=lambda _s: None)
        with self.assertRaises(ProviderError) as caught:
            provider.complete([{"role": "user", "content": "hi"}])
        self.assertIn("not-a-model", str(caught.exception))
        self.assertIn(DOCUMENTED_XAI_MODEL, str(caught.exception))

    def test_empty_response_is_not_saved(self):
        client = FakeClient(response_results=[{"status": "completed", "output_text": "   "}])
        provider = GrokProvider(client, DOCUMENTED_XAI_MODEL, api_key="k", sleep=lambda _s: None)
        memory = RecordingMemory()
        printed = io.StringIO()
        with redirect_stdout(printed):
            history = tinytalk.handle_turn("hello", [], provider, memory, soul="SOUL")
        self.assertEqual(history, [])
        self.assertEqual(memory.saved, [])
        self.assertIn("Nothing was saved", printed.getvalue())


class RoutingAndContextTests(unittest.TestCase):
    def test_grok_request_is_stateless_and_forwards_the_same_messages(self):
        messages = [
            {"role": "system", "content": "soul"},
            {"role": "system", "content": "Explicit user-stated facts:\n\nEnterprise"},
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "now"},
        ]
        client = FakeClient(response_results=[{
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": "answer"}],
            }],
        }])
        provider = GrokProvider(client, DOCUMENTED_XAI_MODEL, api_key="k", sleep=lambda _s: None)
        self.assertEqual(provider.complete(messages), "answer")
        sent = client.response_calls[0]
        self.assertEqual(sent["model"], DOCUMENTED_XAI_MODEL)
        self.assertEqual(sent["input"], messages)
        self.assertIs(sent["store"], False)
        self.assertNotIn("previous_response_id", sent)
        self.assertNotIn("tools", sent)
        self.assertNotIn("tool_choice", sent)
        self.assertNotIn("extra_body", sent)
        self.assertNotIn("search_parameters", sent)
        self.assertNotIn("reasoning", sent)

    def test_ollama_label_and_greeting_follow_the_configured_model(self):
        provider = OllamaProvider(FakeClient(), "qwen2.5:3b", sleep=lambda _s: None)
        self.assertEqual(provider.label, "qwen2.5:3b")
        self.assertIn("qwen2.5:3b", tinytalk.greeting(provider))
        grok = GrokProvider(
            FakeClient(),
            DOCUMENTED_XAI_MODEL,
            api_key="k",
            sleep=lambda _s: None,
        )
        greeting = tinytalk.greeting(grok)
        self.assertIn("Grok", greeting)
        self.assertNotIn("qwen2.5:3b", greeting)

    def test_soul_read_failure_names_the_file(self):
        shown = io.StringIO()
        with redirect_stdout(shown):
            with unittest.mock.patch.object(Path, "read_text", side_effect=OSError("denied")):
                text = tinytalk.load_soul()
        self.assertEqual(text, "You are TinyTalk, a helpful local assistant.")
        self.assertIn("SOUL.md", shown.getvalue())
        self.assertIn("denied", shown.getvalue())

    def test_ollama_forwards_the_same_messages_to_chat_completions(self):
        messages = [
            {"role": "system", "content": "soul"},
            {"role": "user", "content": "hi"},
        ]
        client = FakeClient(chat_results=[{
            "choices": [{"message": {"content": "yo"}}],
        }])
        provider = OllamaProvider(client, DEFAULT_OLLAMA_MODEL, sleep=lambda _s: None)
        self.assertEqual(provider.complete(messages), "yo")
        sent = client.chat_calls[0]
        self.assertEqual(sent["model"], DEFAULT_OLLAMA_MODEL)
        self.assertEqual(sent["messages"], messages)
        self.assertFalse(client.response_calls)

    def test_both_providers_see_soul_memory_and_trimmed_history(self):
        prior = []
        for index in range(12):
            prior.append({"role": "user", "content": "user-%s" % index})
            prior.append({"role": "assistant", "content": "assistant-%s" % index})
        long_reply = "Z" * (tinytalk.MAX_REQUEST_CHARS)
        prior[-1] = {"role": "assistant", "content": long_reply}
        memory = RecordingMemory(facts=["my test spaceship is named Enterprise"])
        soul = "SOUL TEXT"
        scripted = ScriptedProvider(["remembered"])
        updated = tinytalk.handle_turn(
            "What is my test spaceship called?",
            prior,
            scripted,
            memory,
            soul,
        )
        sent = scripted.calls[0]
        self.assertEqual(sent[0], {"role": "system", "content": soul})
        self.assertIn("Enterprise", sent[1]["content"])
        self.assertEqual(sent[1]["role"], "system")
        self.assertEqual(sent[-1], {"role": "user", "content": "What is my test spaceship called?"})
        sent_text = "\n".join(message["content"] for message in sent)
        self.assertNotIn(long_reply, sent_text)
        self.assertNotIn("user-0", sent_text)
        self.assertLessEqual(len(updated), tinytalk.MAX_TURNS * 2)
        self.assertNotIn("Enterprise", updated[-1]["content"])
        self.assertEqual(updated[-1]["content"], "remembered")
        self.assertEqual(memory.saved, [(
            "What is my test spaceship called?",
            "remembered",
        )])

    def test_introspection_uses_the_fact_list(self):
        memory = RecordingMemory(facts=["my test spaceship is named Enterprise"])
        scripted = ScriptedProvider(["I remember the spaceship."])
        tinytalk.handle_turn(
            "What do you remember about me?",
            [],
            scripted,
            memory,
            "SOUL",
        )
        memory_message = scripted.calls[0][1]["content"]
        self.assertIn("saved facts retrieved for this request", memory_message)
        self.assertIn("not a complete inventory of conversation storage", memory_message)
        self.assertIn("merely because they were not included", memory_message)
        self.assertIn("saved memories", memory_message)
        self.assertIn("model training", memory_message)
        self.assertIn("Enterprise", memory_message)
        self.assertTrue(tinytalk.is_memory_introspection("What  do you remember about me?"))
        self.assertFalse(tinytalk.is_memory_introspection("what do you remember about the ship"))

    def test_an_oversized_user_line_is_not_sent(self):
        scripted = ScriptedProvider(["should not be called"])
        memory = RecordingMemory()
        sources = tinytalk.SessionSources()
        sources.replace([{"kind": "saved fact", "text": "keep me"}])
        huge = "x" * tinytalk.MAX_REQUEST_CHARS
        shown = io.StringIO()
        with redirect_stdout(shown):
            updated = tinytalk.handle_turn(huge, [{"role": "user", "content": "old"}], scripted, memory, "SOUL", sources)
        self.assertEqual(scripted.calls, [])
        self.assertEqual(updated, [{"role": "user", "content": "old"}])
        self.assertEqual(memory.saved, [])
        self.assertEqual(sources.records, [{"kind": "saved fact", "text": "keep me"}])
        self.assertIn(str(tinytalk.MAX_REQUEST_CHARS), shown.getvalue())
        self.assertIn("Nothing was saved.", shown.getvalue())

    def test_trailing_records_are_omitted_with_a_count(self):
        facts = ["fact-%s %s" % (index, "z" * 400) for index in range(6)]
        memory = RecordingMemory(facts=facts)
        scripted = ScriptedProvider(["partial"])
        previous = os.environ.get("TINYTALK_MAX_REQUEST_CHARS")
        os.environ["TINYTALK_MAX_REQUEST_CHARS"] = "4000"
        try:
            tinytalk.handle_turn(
                "What do you remember about me?",
                [],
                scripted,
                memory,
                "SOUL",
            )
        finally:
            if previous is None:
                os.environ.pop("TINYTALK_MAX_REQUEST_CHARS", None)
            else:
                os.environ["TINYTALK_MAX_REQUEST_CHARS"] = previous
        self.assertEqual(len(scripted.calls), 1)
        body = scripted.calls[0][1]["content"]
        self.assertIn("fact-0", body)
        self.assertNotIn("fact-5", body)
        self.assertIn("retrieved records were omitted", body)
        self.assertIn("4000", body)

    def test_interrupt_at_the_prompt_stops_speech(self):
        player = StopPlayer()
        speech = SpeechController(object(), player, lambda _message: None)

        def read_line(_prompt):
            raise KeyboardInterrupt

        tinytalk.run_repl(ScriptedProvider(["nope"]), None, "SOUL", speech, None, read_line)
        self.assertGreaterEqual(player.stop_calls, 1)

    def test_interrupt_during_a_turn_stops_speech(self):
        player = StopPlayer()
        speech = SpeechController(object(), player, lambda _message: None)

        class Boom(object):
            label = "Script"

            def complete(self, messages):
                raise KeyboardInterrupt

        def read_line(_prompt):
            return "hello"

        shown = io.StringIO()
        with redirect_stdout(shown):
            updated = tinytalk.run_repl(Boom(), None, "SOUL", speech, None, read_line)
        self.assertEqual(updated, [])
        self.assertGreaterEqual(player.stop_calls, 1)
        self.assertNotIn("Script:", shown.getvalue())

    def test_debug_lines_stay_hidden_unless_enabled(self):
        class Graph(object):
            def query_entity(self, subject):
                return []

            def add_triple(self, *args, **kwargs):
                return None

        class Memory(object):
            palace = object()
            kg = Graph()

            def add_drawer(self, **kwargs):
                return {"success": True}

        provider = ScriptedProvider([
            '{"subject":"user","predicate":"favorite_color","object":"blue"}',
            '{"subject":"user","predicate":"favorite_color","object":"blue"}',
        ])
        previous = os.environ.pop("TINYTALK_DEBUG", None)
        try:
            hidden = io.StringIO()
            with redirect_stdout(hidden):
                tinytalk.remember_fact(
                    "Remember this: my favorite color is blue",
                    Memory(),
                    provider,
                )
            self.assertNotIn("[debug]", hidden.getvalue())

            os.environ["TINYTALK_DEBUG"] = "1"
            shown = io.StringIO()
            with redirect_stdout(shown):
                tinytalk.remember_fact(
                    "Remember this: my favorite color is blue",
                    Memory(),
                    provider,
                )
            self.assertIn("[debug] KG:", shown.getvalue())
        finally:
            if previous is None:
                os.environ.pop("TINYTALK_DEBUG", None)
            else:
                os.environ["TINYTALK_DEBUG"] = previous

    def test_failed_remember_does_not_extract_or_save(self):
        scripted = ScriptedProvider([
            ProviderError("xAI rejected the API key. TinyTalk did not switch to another provider."),
        ])
        memory = RecordingMemory()
        memory.kg = object()
        memory.add_drawer = lambda **kwargs: (_ for _ in ()).throw(AssertionError("saved"))
        history = tinytalk.handle_turn(
            "Remember this: my test spaceship is named Enterprise",
            [{"role": "user", "content": "old"}, {"role": "assistant", "content": "old reply"}],
            scripted,
            memory,
            "SOUL",
        )
        self.assertEqual(len(scripted.calls), 1)
        self.assertEqual(history, [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old reply"},
        ])
        self.assertEqual(memory.saved, [])

    def test_fact_extraction_uses_the_provider_and_keeps_its_contract(self):
        scripted = ScriptedProvider([
            'Sure. {"subject":"my","predicate":"test_spaceship","object":"Enterprise"}'
        ])
        triple = tinytalk.fact_to_triple(
            "my test spaceship is named Enterprise",
            scripted,
        )
        self.assertEqual(triple, ("user", "test_spaceship_name", "Enterprise"))
        self.assertEqual(scripted.calls[0][0]["role"], "system")
        self.assertIn("subject", scripted.calls[0][0]["content"])
        self.assertNotIn("SOUL", scripted.calls[0][0]["content"])

        empty = ScriptedProvider(['{"subject":"","predicate":"","object":""}'])
        self.assertIsNone(tinytalk.fact_to_triple("not a fact", empty))


class SdkRetryTests(unittest.TestCase):
    def test_repeated_429_and_500_make_exactly_three_http_requests(self):
        """The real OpenAI client, with a mocked transport, must not add retries."""
        cases = (
            ("ollama", 429, "/chat/completions"),
            ("ollama", 500, "/chat/completions"),
            ("grok", 429, "/responses"),
            ("grok", 500, "/responses"),
        )
        for provider_name, status, path_suffix in cases:
            seen = []

            def handler(request, status=status, seen=seen):
                seen.append(request)
                return httpx.Response(
                    status,
                    json={"error": {"message": "unavailable", "type": "api_error"}},
                )

            transport = httpx.MockTransport(handler)
            with httpx.Client(transport=transport) as http_client:
                if provider_name == "ollama":
                    settings = load_settings({"TINYTALK_PROVIDER": "ollama"})
                else:
                    settings = load_settings({
                        "TINYTALK_PROVIDER": "grok",
                        "XAI_API_KEY": "test-key",
                        "XAI_MODEL": DOCUMENTED_XAI_MODEL,
                    })
                provider = build_provider(
                    settings,
                    sleep=lambda _delay: None,
                    http_client=http_client,
                )
                with self.assertRaises(ProviderError):
                    provider.complete([{"role": "user", "content": "hi"}])

            self.assertEqual(
                len(seen),
                3,
                "%s HTTP %s sent %s requests" % (provider_name, status, len(seen)),
            )
            for request in seen:
                self.assertTrue(request.url.path.endswith(path_suffix), request.url.path)
            if provider_name == "grok":
                for request in seen:
                    body = json.loads(request.content.decode("utf-8"))
                    self.assertIs(body["store"], False)
                    self.assertNotIn("tools", body)
                    self.assertNotIn("tool_choice", body)
                    self.assertNotIn("search_parameters", body)
                    self.assertNotIn("previous_response_id", body)


class DotEnvTests(unittest.TestCase):
    def test_dotenv_does_not_override_existing_values(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".env"
            path.write_text(
                "TINYTALK_PROVIDER=ollama\nXAI_API_KEY=from-file\n# comment\n",
                encoding="utf-8",
            )
            previous_provider = os.environ.get("TINYTALK_PROVIDER")
            previous_key = os.environ.get("XAI_API_KEY")
            os.environ["TINYTALK_PROVIDER"] = "grok"
            os.environ.pop("XAI_API_KEY", None)
            try:
                tinytalk.load_dotenv(path)
                self.assertEqual(os.environ["TINYTALK_PROVIDER"], "grok")
                self.assertEqual(os.environ["XAI_API_KEY"], "from-file")
            finally:
                if previous_provider is None:
                    os.environ.pop("TINYTALK_PROVIDER", None)
                else:
                    os.environ["TINYTALK_PROVIDER"] = previous_provider
                if previous_key is None:
                    os.environ.pop("XAI_API_KEY", None)
                else:
                    os.environ["XAI_API_KEY"] = previous_key


if __name__ == "__main__":
    unittest.main()
