"""Session source labels. No live model and no real MemPalace."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tinytalk
from provider import ProviderError


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
    def __init__(self):
        self.calls = []

    def speak_last(self, messages):
        self.calls.append("speak")

    def stop(self):
        self.calls.append("stop")


class SwitchingMemory(object):
    palace = object()
    kg = None

    def __init__(self):
        self.saved = []
        self.mode = "facts"

    def search_facts(self, query):
        if self.mode != "facts":
            return []
        return [{
            "text": "my test spaceship is named Enterprise",
            "drawer_id": "drawer_enterprise",
            "room": "facts",
            "filed_at": "2026-09-27T23:08:05",
            "similarity": 0.8,
        }]

    def search_conversations(self, query):
        if self.mode != "conversation":
            return []
        return [{
            "text": "User: hello there\n\nAssistant: hi",
            "drawer_id": "drawer_convo",
            "filed_at": "unknown",
            "similarity": 0.8,
        }]

    def save_exchange(self, user_prompt, response):
        self.saved.append((user_prompt, response))

    def list_facts(self):
        raise AssertionError("list_facts")


class SourceTests(unittest.TestCase):
    def _turn(self, memory, provider, sources, prompt):
        printed = io.StringIO()
        with redirect_stdout(printed):
            tinytalk.handle_turn(prompt, [], provider, memory, "SOUL", sources)
        self.assertNotIn("These saved memories were included", printed.getvalue())

    def test_saved_fact_and_conversation_labels_keep_available_metadata(self):
        memory = SwitchingMemory()
        sources = tinytalk.SessionSources()
        provider = ScriptedProvider(["Enterprise.", "From the chat."])
        self._turn(memory, provider, sources, "What is my test spaceship called?")
        prompt = provider.calls[0][1]["content"]
        self.assertIn("[saved fact]", prompt)
        self.assertIn("id: drawer_enterprise", prompt)
        self.assertIn("my test spaceship is named Enterprise", prompt)
        self.assertIn("filed: 2026-09-27T23:08:05", prompt)
        self.assertIn("not necessarily when the fact changed", prompt)
        self.assertIn("retrieved source", prompt)
        saved = next(
            record for record in sources.records
            if record.get("selection") == "semantic fact"
        )
        self.assertEqual(saved["kind"], "saved fact")
        self.assertEqual(saved["id"], "drawer_enterprise")
        self.assertEqual(saved["status"], "current")

        report = io.StringIO()
        with redirect_stdout(report):
            updated = tinytalk.handle_user_line(
                "/sources",
                [{"role": "assistant", "content": "Enterprise."}],
                provider,
                memory,
                "SOUL",
                FakeSpeech(),
                sources,
            )
        text = report.getvalue()
        self.assertIn("saved fact", text)
        self.assertIn("drawer_enterprise", text)
        self.assertIn("my test spaceship is named Enterprise", text)
        self.assertIn("included in the context", text)
        self.assertIn("does not claim the model used every one", text)
        self.assertEqual(updated[-1]["content"], "Enterprise.")
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(len(memory.saved), 1)

        memory.mode = "conversation"
        printed = io.StringIO()
        with redirect_stdout(printed):
            tinytalk.handle_user_line(
                "/recall what did we say?",
                [],
                provider,
                memory,
                "SOUL",
                FakeSpeech(),
                sources,
            )
        convo = provider.calls[-1][1]["content"]
        self.assertIn("[conversation excerpt]", convo)
        self.assertIn("explicit history recall", convo)
        self.assertIn("status: historical", convo)
        self.assertIn("id: drawer_convo", convo)
        self.assertIn("User: hello there\n\nAssistant: hi", convo)
        self.assertNotIn("filed:", convo)
        excerpt = next(
            record for record in sources.records
            if record["kind"] == "conversation excerpt"
        )
        self.assertEqual(excerpt["selection"], "explicit history recall")
        self.assertEqual(excerpt["status"], "historical")
        self.assertNotIn("filed_at", excerpt)

    def test_historical_sources_exclude_the_imaginary_spaceship(self):
        class Memory(object):
            palace = object()

            def __init__(self):
                self.saved = []
                self.kg = self

            def query_entity(self, subject):
                return [
                    {
                        "subject": "user",
                        "predicate": "test_spaceship_name",
                        "object": "Enterprise",
                        "current": True,
                        "valid_from": None,
                        "valid_to": None,
                    },
                    {
                        "subject": "user",
                        "predicate": "test_spaceship",
                        "object": "Enterprise",
                        "current": False,
                        "valid_from": None,
                        "valid_to": "2026-10-03",
                    },
                ]

            def list_drawers(self, wing, room, limit, offset=0):
                if room == "facts":
                    return {"drawers": [
                        {
                            "drawer_id": "drawer_enterprise",
                            "room": "facts",
                            "content_preview": "my test spaceship is named Enterprise",
                            "metadata": {"filed_at": "2026-09-27T23:08:05", "room": "facts"},
                        },
                        {
                            "drawer_id": "drawer_pickle",
                            "room": "facts",
                            "content_preview": "the name of my imaginary spaceship is Picklewagon",
                            "metadata": {"filed_at": "2026-09-27T20:54:31", "room": "facts"},
                        },
                    ]}
                return {"drawers": [
                    {
                        "drawer_id": "drawer_serenity",
                        "room": "facts-history",
                        "content_preview": "my test spaceship is named Serenity.",
                        "metadata": {"filed_at": "2026-09-27T22:17:31", "room": "facts-history"},
                    },
                ]}

            def search_facts(self, query):
                raise AssertionError(query)

            def search_conversations(self, query):
                raise AssertionError(query)

            def save_exchange(self, user_prompt, response):
                self.saved.append((user_prompt, response))

        memory = Memory()
        sources = tinytalk.SessionSources()
        provider = ScriptedProvider(["Serenity."])
        self._turn(
            memory,
            provider,
            sources,
            "What was the previous name of my test spaceship?",
        )
        kinds = [record["kind"] for record in sources.records]
        self.assertIn("saved fact", kinds)
        self.assertIn("archived fact", kinds)
        self.assertIn("knowledge-graph record", kinds)
        archived = [record for record in sources.records if record["kind"] == "archived fact"]
        self.assertEqual(archived[0]["id"], "drawer_serenity")
        self.assertEqual(archived[0]["text"], "my test spaceship is named Serenity.")
        self.assertEqual(archived[0]["filed_at"], "2026-09-27T22:17:31")
        graph = [
            record for record in sources.records
            if record["kind"] == "knowledge-graph record" and record.get("valid_to")
        ]
        self.assertEqual(graph[0]["valid_to"], "2026-10-03")
        self.assertNotIn("id", graph[0])
        blob = "\n".join(record["text"] for record in sources.records)
        self.assertNotIn("Picklewagon", blob)
        prompt = provider.calls[0][1]["content"]
        self.assertIn("[archived fact]", prompt)
        self.assertIn("[knowledge-graph record]", prompt)
        self.assertIn("not necessarily when the fact changed", prompt)
        self.assertIn("Do not give a rename date.", prompt)

    def test_empty_retrieval_clears_sources_and_a_failure_keeps_them(self):
        memory = SwitchingMemory()
        sources = tinytalk.SessionSources()
        provider = ScriptedProvider([
            "Enterprise.",
            ProviderError("the model failed"),
            "No memory in this one.",
        ])
        self._turn(memory, provider, sources, "What is my test spaceship called?")
        kept = list(sources.records)
        self.assertTrue(any(record.get("id") == "drawer_enterprise" for record in kept))
        printed = io.StringIO()
        with redirect_stdout(printed):
            history = tinytalk.handle_turn(
                "hello",
                [],
                provider,
                memory,
                "SOUL",
                sources,
            )
        self.assertEqual(history, [])
        self.assertEqual(sources.records, kept)
        self.assertEqual(len(memory.saved), 1)

        memory.mode = "empty"
        self._turn(memory, provider, sources, "Something else")
        self.assertFalse(any(record.get("id") == "drawer_enterprise" for record in sources.records))
        self.assertTrue(sources.records)
        self.assertTrue(all(record.get("status") == "unavailable" for record in sources.records))
        report = tinytalk.format_sources_report([])
        self.assertIn("No saved memories were included", report)
        self.assertIn("does not mean no memories exist", report)
        self.assertIn("does not mean the answer came only from model knowledge", report)
        self.assertNotIn("No memories exist.", report)

    def test_sources_and_speech_do_not_call_the_model_or_save(self):
        class Memory(object):
            palace = object()
            kg = None

            def save_exchange(self, *args):
                raise AssertionError("saved")

            def search_facts(self, query):
                raise AssertionError(query)

            def search_conversations(self, query):
                raise AssertionError(query)

        class Provider(object):
            label = "Script"

            def complete(self, messages):
                raise AssertionError("model")

        memory = Memory()
        sources = tinytalk.SessionSources()
        sources.replace([
            tinytalk._source(
                "saved fact",
                "my test spaceship is named Enterprise",
                "drawer_enterprise",
                filed_at="2026-09-27T23:08:05",
            )
        ])
        speech = FakeSpeech()
        messages = [{"role": "assistant", "content": "Enterprise."}]
        for command in ("/sources", "/speak", "/stop"):
            updated = tinytalk.handle_user_line(
                command,
                messages,
                Provider(),
                memory,
                "SOUL",
                speech,
                sources,
            )
            self.assertEqual(updated, messages)
            self.assertEqual(sources.records[0]["id"], "drawer_enterprise")
        self.assertEqual(speech.calls, ["speak", "stop"])


class ConversationFloorTests(unittest.TestCase):
    def _memory(self, hits, history=()):
        class Memory(object):
            palace = object()
            kg = None

            def search_facts(self, query):
                return []

            def search_conversations(self, query):
                return hits

            def unfinished_updates(self):
                return []

            def list_drawers(self, wing, room, limit=100, offset=0):
                if room != "facts-history":
                    return {"drawers": [], "total": 0}
                drawers = [
                    {
                        "drawer_id": "history-%s" % index,
                        "content_preview": text,
                        "metadata": {},
                    }
                    for index, text in enumerate(history)
                ]
                return {"drawers": drawers[offset:offset + limit], "total": len(drawers)}

        return Memory()

    def _texts(self, hits, history=()):
        _messages, sources = tinytalk.recall_context(
            "what did we say",
            [{"role": "user", "content": "/recall what did we say"}],
            self._memory(hits, history),
        )
        return [source["text"] for source in sources if source["kind"] == "conversation excerpt"]

    def test_a_low_similarity_excerpt_is_left_out(self):
        hits = [{
            "text": "User: Remember this: call me Al",
            "similarity": 0.01,
        }]
        self.assertEqual(self._texts(hits), [])

    def test_a_replaced_command_is_left_out(self):
        hits = [{
            "text": "User: Remember this: call me Al\n\nAssistant: Okay.",
            "similarity": 0.8,
        }]
        self.assertEqual(self._texts(hits, history=("call me Al",)), [])

    def test_ordinary_chatter_above_the_floor_is_included(self):
        hits = [{
            "text": "User: hello\n\nAssistant: hi",
            "similarity": 0.8,
        }]
        self.assertEqual(self._texts(hits), ["User: hello\n\nAssistant: hi"])


if __name__ == "__main__":
    unittest.main()
