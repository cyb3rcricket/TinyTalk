"""Gradio UI over handle_user_line. These tests never launch a server."""

import io
import json
import os
import sys
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory_isolation  # noqa: E402  (must come before tinytalk and mempalace)
import gradio as gr
import tinytalk
import gradio_app
from provider import ConfigError


def triple(predicate, obj, subject="user"):
    return json.dumps({"subject": subject, "predicate": predicate, "object": obj})


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


class TextProvider(object):
    """Extraction stub keyed by the fact text. Chat replies use `reply`."""

    name = "stub"
    label = "Stub"

    def __init__(self, answers=None, reply="Okay."):
        self.answers = dict(answers or {})
        self.reply = reply
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages)
        text = messages[-1]["content"]
        if messages[0]["content"] != tinytalk.TRIPLE_INSTRUCTIONS:
            return self.reply
        answer = self.answers.get(text, "not json")
        if isinstance(answer, Exception):
            raise answer
        return answer


def _dependency(app, name):
    for dep in app.fns.values():
        if dep.fn is not None and getattr(dep.fn, "__name__", "") == name:
            return dep
    raise AssertionError("no wired handler named %s" % name)


def _submit(app, text, messages, sources, chat=None):
    return _dependency(app, "submit_message").fn(text, chat, messages, sources)


class GradioAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = memory_isolation.temp_dir("tinytalk-gradio-")
        self.memory = memory_isolation.open_isolated_memory(self.temp.name)

    def tearDown(self):
        self.memory.kg.close()
        self.temp.cleanup()

    def _app(self, provider, sources=None):
        return gradio_app.create_app(
            provider=provider,
            memory=self.memory,
            soul="SOUL",
            sources=sources,
        )

    def test_create_app_returns_blocks_without_launching(self):
        provider = ScriptedProvider(["unused"])
        app = self._app(provider)
        self.assertIsInstance(app, gr.Blocks)
        self.assertFalse(app.is_running)
        self.assertIsNone(app.local_url)
        self.assertFalse(app.share)
        self.assertFalse(hasattr(app, "server"))
        self.assertEqual(app.title, "TinyTalk")
        self.assertFalse(app.analytics_enabled)
        self.assertIsNone(app.css)
        kinds = [type(block).__name__ for block in app.blocks.values()]
        self.assertIn("Chatbot", kinds)
        self.assertIn("Textbox", kinds)
        self.assertIn("Button", kinds)
        self.assertIn("State", kinds)
        chatbots = [block for block in app.blocks.values() if type(block).__name__ == "Chatbot"]
        self.assertEqual(len(chatbots), 1)
        self.assertNotIn("share", list(chatbots[0].buttons or []))
        self.assertEqual(chatbots[0].avatar_images, [None, None])
        self.assertTrue(any("TinyTalk" in str(getattr(block, "value", "")) for block in app.blocks.values()))
        buttons = [block for block in app.blocks.values() if type(block).__name__ == "Button"]
        self.assertTrue(any(getattr(block, "value", None) == "Clear" for block in buttons))
        submit = _dependency(app, "submit_message")
        self.assertIn("submit", [target[1] for target in submit.targets])
        clear = _dependency(app, "clear_conversation")
        self.assertIn("click", [target[1] for target in clear.targets])
        self.assertEqual(provider.calls, [])

    def test_normal_turn_shows_the_reply_once_without_label_or_separator(self):
        provider = ScriptedProvider(["The castle is quiet."])
        app = self._app(provider)
        sources = tinytalk.SessionSources()
        chat, messages, _sources, box = _submit(app, "Hello", [], sources)
        self.assertEqual(chat, [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "The castle is quiet."},
        ])
        self.assertEqual(messages[-1]["content"], "The castle is quiet.")
        self.assertEqual(box, "")
        self.assertEqual(provider.calls[0][-1]["content"], "Hello")

    def test_blank_submission_is_ignored(self):
        provider = ScriptedProvider(["should not be used"])
        app = self._app(provider)
        sources = tinytalk.SessionSources()
        existing = [{"role": "assistant", "content": "earlier"}]
        chat, messages, returned, box = _submit(app, "   ", existing, sources)
        self.assertEqual(messages, existing)
        self.assertEqual(chat, [{"role": "assistant", "content": "earlier"}])
        self.assertIs(returned, sources)
        self.assertEqual(box, "")
        self.assertEqual(provider.calls, [])
        shown = [
            {"role": "user", "content": "/memory"},
            {"role": "assistant", "content": "Current profile"},
        ]
        kept, messages, returned, box = _submit(app, " \n ", existing, sources, chat=shown)
        self.assertEqual(kept, shown)
        self.assertIs(messages, existing)
        self.assertIs(returned, sources)
        self.assertEqual(box, "")
        self.assertEqual(provider.calls, [])

    def test_remember_turn_shows_the_memory_line_and_stores_the_fact(self):
        fact = "my home city is Austin"
        provider = TextProvider(
            {fact: triple("home_city", "Austin")},
            reply="I will remember Austin.",
        )
        app = self._app(provider)
        sources = tinytalk.SessionSources()
        chat, messages, _sources, _box = _submit(
            app, "Remember this: " + fact, [], sources
        )
        self.assertEqual(chat[0], {"role": "user", "content": "Remember this: " + fact})
        self.assertEqual(chat[-1]["role"], "assistant")
        self.assertEqual(chat[-1]["content"].count("I will remember Austin."), 1)
        self.assertIn("\n\nMemory: saved.", chat[-1]["content"])
        self.assertNotIn("Stub:", chat[-1]["content"])
        self.assertNotIn("-" * 30, chat[-1]["content"])
        self.assertEqual(messages[-1]["content"], "I will remember Austin.")
        self.assertTrue(any(fact in saved for saved in self.memory.list_facts()))

    def test_memory_and_forget_report_the_profile(self):
        fact = "my home city is Austin"
        provider = TextProvider({fact: triple("home_city", "Austin")}, reply="Noted.")
        app = self._app(provider)
        sources = tinytalk.SessionSources()
        _chat, messages, sources, _box = _submit(
            app, "Remember this: " + fact, [], sources
        )
        before = list(messages)
        chat, messages, sources, _box = _submit(app, "/memory", messages, sources)
        report = tinytalk.format_memory_report(tinytalk.read_current_profile(self.memory))
        self.assertEqual(messages, before)
        self.assertEqual(chat[-2], {"role": "user", "content": "/memory"})
        self.assertEqual(chat[-1], {"role": "assistant", "content": report})
        self.assertIn("home_city: current", report)
        self.assertIn("value: Austin", report)

        chat, messages, sources, _box = _submit(app, "/forget home_city", messages, sources)
        self.assertEqual(messages, before)
        self.assertEqual(chat[-2], {"role": "user", "content": "/forget home_city"})
        self.assertIn("Forgot facts for home_city.", chat[-1]["content"])
        self.assertNotIn("-" * 30, chat[-1]["content"])

        profile = {
            entry["predicate"]: entry
            for entry in tinytalk.read_current_profile(self.memory)
        }
        self.assertEqual(profile["home_city"]["state"], "missing")
        self.assertIsNone(profile["home_city"]["value"])
        chat, messages, _sources, _box = _submit(app, "/memory", messages, sources)
        shown = chat[-1]["content"]
        self.assertIn("home_city: missing", shown)
        self.assertNotIn("value: Austin", shown)
        self.assertEqual(messages, before)

    def test_clear_resets_conversation_state(self):
        provider = ScriptedProvider(["Hi"])
        sources = tinytalk.SessionSources()
        app = self._app(provider, sources)
        _chat, messages, sources, _box = _submit(app, "Hello", [], sources)
        sources.replace([{"kind": "saved fact", "text": "kept only for this session"}])
        self.assertTrue(messages)
        cleared = _dependency(app, "clear_conversation").fn()
        self.assertEqual(cleared[0], [])
        self.assertEqual(cleared[1], [])
        self.assertEqual(cleared[2].records, [])
        self.assertIsNot(cleared[2], sources)
        self.assertEqual(cleared[3], "")

    def test_main_launches_on_localhost_without_sharing(self):
        app = unittest.mock.Mock()
        with unittest.mock.patch("gradio_app.create_app", return_value=app):
            with unittest.mock.patch.dict(os.environ, {"TINYTALK_WEB_PORT": "7877"}):
                code = gradio_app.main()
        app.launch.assert_called_once_with(
            share=False, server_name="127.0.0.1", server_port=7877
        )
        self.assertEqual(code, 0)

        app.launch.reset_mock()
        env = os.environ.copy()
        env.pop("TINYTALK_WEB_PORT", None)
        with unittest.mock.patch("gradio_app.create_app", return_value=app):
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                code = gradio_app.main()
        app.launch.assert_called_once_with(
            share=False, server_name="127.0.0.1", server_port=7861
        )
        self.assertEqual(code, 0)

    def test_main_reports_configuration_errors(self):
        shown = io.StringIO()
        with unittest.mock.patch(
            "gradio_app.create_app",
            side_effect=ConfigError("Grok mode needs XAI_API_KEY"),
        ):
            with redirect_stdout(shown):
                code = gradio_app.main()
        self.assertEqual(code, 1)
        self.assertIn("Configuration error: Grok mode needs XAI_API_KEY", shown.getvalue())

    def test_main_falls_back_to_default_port_on_invalid_env(self):
        app = unittest.mock.Mock()
        for bad in ("", "not-a-port", "0", "70000"):
            app.launch.reset_mock()
            with unittest.mock.patch("gradio_app.create_app", return_value=app):
                with unittest.mock.patch.dict(os.environ, {"TINYTALK_WEB_PORT": bad}):
                    code = gradio_app.main()
            app.launch.assert_called_once_with(
                share=False, server_name="127.0.0.1", server_port=7861
            )
            self.assertEqual(code, 0)

    def test_command_bubbles_survive_subsequent_chat_turns(self):
        provider = ScriptedProvider(["Hello again."])
        app = self._app(provider)
        sources = tinytalk.SessionSources()
        chat, messages, sources, _box = _dependency(app, "submit_message").fn(
            "/memory", [], [], sources
        )
        self.assertEqual(len(chat), 2)
        self.assertEqual(chat[0]["content"], "/memory")
        self.assertIn("Current profile", chat[1]["content"])

        chat, messages, sources, _box = _dependency(app, "submit_message").fn(
            "Hi", chat, messages, sources
        )
        self.assertEqual(len(chat), 4)
        self.assertEqual(chat[0]["content"], "/memory")
        self.assertIn("Current profile", chat[1]["content"])
        self.assertEqual(chat[2], {"role": "user", "content": "Hi"})
        self.assertEqual(chat[3], {"role": "assistant", "content": "Hello again."})

    def test_submit_does_not_mutate_debug_env_or_leak_debug(self):
        fact = "my home city is Austin"
        provider = TextProvider({fact: triple("home_city", "Austin")}, reply="Noted.")
        app = self._app(provider)
        sources = tinytalk.SessionSources()
        with unittest.mock.patch.dict(os.environ, {"TINYTALK_DEBUG": "1"}):
            chat, _messages, _sources, _box = _submit(
                app, "Remember this: " + fact, [], sources
            )
            self.assertEqual(os.environ["TINYTALK_DEBUG"], "1")
        self.assertNotIn("[debug]", chat[-1]["content"])
