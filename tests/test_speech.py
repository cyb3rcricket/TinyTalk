"""VoiceStudio speech commands. No live server and no real MemPalace."""

import json
import os
import sys
import threading
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tinytalk
from speech import (
    DEFAULT_VOICESTUDIO_SEED,
    DEFAULT_VOICESTUDIO_VOICE_ID,
    SpeechController,
    SpeechError,
    VoiceStudioSpeech,
    last_assistant_answer,
)


class BlockingClient(object):
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.error = None
        self.audio = b"RIFF-fake-wav"

    def synthesize(self, text):
        self.calls.append(text)
        self.started.set()
        if not self.release.wait(2):
            raise SpeechError("timed out in test")
        if self.error:
            raise SpeechError(self.error)
        return self.audio


class ScriptedPlayer(object):
    def __init__(self):
        self.plays = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.stop_calls = 0

    def play(self, wav_bytes, still_current):
        self.plays.append(wav_bytes)
        self.started.set()
        while not self.release.is_set():
            if not still_current():
                return False
            self.release.wait(0.01)
        return bool(still_current())

    def stop(self):
        self.stop_calls += 1
        self.release.set()


class RecordingOutput(object):
    def __init__(self):
        self.lines = []
        self.lock = threading.Lock()

    def __call__(self, message):
        with self.lock:
            self.lines.append(message)

    def joined(self):
        with self.lock:
            return "\n".join(self.lines)


class SpeechCommandTests(unittest.TestCase):
    def test_request_uses_the_saved_profile_and_ignores_the_xai_key(self):
        seen = []
        secret = "xai-test-key-should-not-leave-tinytalk"

        def handler(request):
            seen.append(request)
            return httpx.Response(200, content=b"RIFF-ok")

        previous = os.environ.get("XAI_API_KEY")
        os.environ["XAI_API_KEY"] = secret
        try:
            client = VoiceStudioSpeech(
                "http://127.0.0.1:3900",
                DEFAULT_VOICESTUDIO_VOICE_ID,
                transport=httpx.MockTransport(handler),
            )
            audio = client.synthesize("Hello from TinyTalk.")
        finally:
            if previous is None:
                os.environ.pop("XAI_API_KEY", None)
            else:
                os.environ["XAI_API_KEY"] = previous

        self.assertEqual(audio, b"RIFF-ok")
        self.assertEqual(len(seen), 1)
        request = seen[0]
        self.assertEqual(request.url.path, "/v1/audio/speech")
        self.assertNotIn("authorization", {k.lower() for k in request.headers})
        raw = request.content.decode("utf-8")
        self.assertNotIn(secret, raw)
        self.assertNotIn("XAI_API_KEY", raw)
        body = json.loads(raw)
        self.assertEqual(body["model"], "omnivoice")
        self.assertEqual(body["voice"], "5e49413d")
        self.assertNotEqual(body["voice"], "default")
        self.assertEqual(body["seed"], DEFAULT_VOICESTUDIO_SEED)
        self.assertEqual(body["response_format"], "wav")
        self.assertEqual(body["input"], "Hello from TinyTalk.")
        self.assertNotIn("instruct", body)

    def test_http_failure_is_not_retried(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(503, json={"detail": "busy"})

        client = VoiceStudioSpeech(
            "http://127.0.0.1:3900",
            "5e49413d",
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaises(SpeechError) as caught:
            client.synthesize("Hello")
        self.assertIn("HTTP 503", str(caught.exception))
        self.assertEqual(len(calls), 1)

    def test_success_plays_the_last_assistant_answer_only(self):
        client = BlockingClient()
        client.release.set()
        player = ScriptedPlayer()
        player.release.set()
        output = RecordingOutput()
        speech = SpeechController(client, player, output)
        messages = [
            {"role": "user", "content": "Remember this: secret fact"},
            {"role": "assistant", "content": "I'll remember that."},
        ]
        provider_calls = []

        def complete(messages_for_model):
            provider_calls.append(messages_for_model)
            return "should not be called"

        updated = tinytalk.handle_user_line(
            "/speak",
            messages,
            type("P", (), {"complete": staticmethod(complete), "label": "Script"})(),
            None,
            "SOUL",
            speech,
        )
        speech.join(2)
        self.assertEqual(updated, messages)
        self.assertEqual(provider_calls, [])
        self.assertEqual(client.calls, ["I'll remember that."])
        self.assertEqual(player.plays, [b"RIFF-fake-wav"])
        self.assertIn("Speaking the last answer…", output.joined())
        self.assertIn("Played the last answer.", output.joined())
        self.assertIsNone(last_assistant_answer([
            {"role": "system", "content": '{"subject":"user","predicate":"x","object":"y"}'},
        ]))

    def test_failure_is_reported_and_does_not_play(self):
        client = BlockingClient()
        client.release.set()
        client.error = "Could not reach VoiceStudio at http://127.0.0.1:3900."
        player = ScriptedPlayer()
        player.release.set()
        output = RecordingOutput()
        speech = SpeechController(client, player, output)
        speech.speak_last([{"role": "assistant", "content": "Hello"}])
        speech.join(2)
        self.assertEqual(player.plays, [])
        self.assertIn("Text chat still works.", output.joined())
        self.assertNotIn("Played the last answer.", output.joined())

    def test_a_second_speak_does_not_queue(self):
        client = BlockingClient()
        player = ScriptedPlayer()
        output = RecordingOutput()
        speech = SpeechController(client, player, output)
        messages = [{"role": "assistant", "content": "First answer"}]
        speech.speak_last(messages)
        self.assertTrue(client.started.wait(1))
        speech.speak_last([{"role": "assistant", "content": "Second answer"}])
        client.release.set()
        player.release.set()
        speech.join(2)
        self.assertEqual(client.calls, ["First answer"])
        self.assertIn("did not start another clip", output.joined())

    def test_stop_suppresses_playback_of_a_pending_generation(self):
        client = BlockingClient()
        player = ScriptedPlayer()
        output = RecordingOutput()
        speech = SpeechController(client, player, output)
        speech.speak_last([{"role": "assistant", "content": "Wait for me"}])
        self.assertTrue(client.started.wait(1))
        speech.stop()
        client.release.set()
        speech.join(2)
        self.assertEqual(player.plays, [])
        text = output.joined()
        self.assertIn("Stopped playback.", text)
        self.assertIn("may still finish generating", text)
        self.assertNotIn("Played the last answer.", text)

    def test_stop_ends_playback_already_in_progress(self):
        client = BlockingClient()
        client.release.set()
        player = ScriptedPlayer()
        output = RecordingOutput()
        speech = SpeechController(client, player, output)
        speech.speak_last([{"role": "assistant", "content": "Play this"}])
        self.assertTrue(player.started.wait(1))
        speech.stop()
        speech.join(2)
        self.assertEqual(len(player.plays), 1)
        self.assertGreaterEqual(player.stop_calls, 1)
        text = output.joined()
        self.assertIn("Stopped playback.", text)
        self.assertNotIn("may still finish generating", text)
        self.assertNotIn("Played the last answer.", text)

    def test_chat_text_is_not_a_speech_command(self):
        client = BlockingClient()
        client.release.set()
        player = ScriptedPlayer()
        output = RecordingOutput()
        speech = SpeechController(client, player, output)

        class Provider(object):
            label = "Script"
            calls = []

            def complete(self, messages):
                self.calls.append(messages)
                return "Typed reply"

        provider = Provider()
        updated = tinytalk.handle_user_line(
            "hello",
            [],
            provider,
            None,
            "SOUL",
            speech,
        )
        self.assertEqual(client.calls, [])
        self.assertEqual(provider.calls[0][-1]["content"], "hello")
        self.assertEqual(updated[-1]["content"], "Typed reply")
        speech.speak_last(updated)
        speech.join(2)
        self.assertEqual(client.calls, ["Typed reply"])


if __name__ == "__main__":
    unittest.main()
