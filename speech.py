"""Optional VoiceStudio playback for TinyTalk.

This client talks only to the local VoiceStudio speech endpoint. It does not
read XAI_API_KEY, does not retry, and does not call the chat provider.
"""

import os
import signal
import subprocess
import tempfile
import threading

import httpx

DEFAULT_VOICESTUDIO_BASE_URL = "http://127.0.0.1:3900"
DEFAULT_VOICESTUDIO_VOICE_ID = "5e49413d"
# Seed stored on the Curious Little Computer design profile.
DEFAULT_VOICESTUDIO_SEED = 42
SPEECH_MODEL = "omnivoice"
# The installed API rejects input longer than this.
MAX_SPEECH_CHARS = 4096
SPEECH_TIMEOUT = httpx.Timeout(300.0, connect=10.0)


class SpeechError(Exception):
    """VoiceStudio could not produce audio. Text chat can continue."""


def speech_settings(env):
    """Backend URL and profile id. Missing values use the local installation."""
    base_url = (env.get("VOICESTUDIO_BASE_URL") or DEFAULT_VOICESTUDIO_BASE_URL).strip()
    if not base_url:
        base_url = DEFAULT_VOICESTUDIO_BASE_URL
    voice_id = (env.get("VOICESTUDIO_VOICE_ID") or DEFAULT_VOICESTUDIO_VOICE_ID).strip()
    if not voice_id:
        voice_id = DEFAULT_VOICESTUDIO_VOICE_ID
    return base_url.rstrip("/"), voice_id


def last_assistant_answer(messages):
    """The latest user-visible assistant reply, if there is one.

    Fact-extraction calls are not stored in this list, so they cannot be spoken.
    """
    for message in reversed(messages or []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def command_kind(text):
    """Return speak, stop, usage, or None. None means ordinary chat."""
    lowered = (text or "").strip().lower()
    if lowered == "/speak":
        return "speak"
    if lowered == "/stop":
        return "stop"
    if lowered.startswith("/speak") or lowered.startswith("/stop"):
        return "usage"
    return None


class VoiceStudioSpeech(object):
    """One POST /v1/audio/speech. No retries and no chat credentials."""

    def __init__(self, base_url, voice_id, seed=DEFAULT_VOICESTUDIO_SEED,
                 timeout=None, transport=None):
        self.base_url = base_url.rstrip("/")
        self.voice_id = voice_id
        self.seed = seed
        self.timeout = timeout or SPEECH_TIMEOUT
        self._transport = transport

    def synthesize(self, text):
        if not isinstance(text, str) or not text.strip():
            raise SpeechError("There is no assistant answer to speak.")
        if len(text) > MAX_SPEECH_CHARS:
            raise SpeechError(
                "That answer is longer than VoiceStudio accepts "
                "(%s characters)." % MAX_SPEECH_CHARS
            )
        # voice is the saved profile id. The installed API resolves that id
        # to the profile's reference clip, transcript, and instruct. OpenAI
        # voice names and "default" skip that lookup. seed is the saved seed;
        # the server would also copy it from the profile when omitted.
        payload = {
            "model": SPEECH_MODEL,
            "input": text,
            "voice": self.voice_id,
            "response_format": "wav",
            "seed": self.seed,
            "speed": 1.0,
            "language": "en",
        }
        url = self.base_url + "/v1/audio/speech"
        try:
            with httpx.Client(
                timeout=self.timeout,
                transport=self._transport,
                trust_env=False,
            ) as client:
                response = client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
        except httpx.TimeoutException:
            raise SpeechError(
                "VoiceStudio timed out at %s." % self.base_url
            )
        except httpx.HTTPError:
            raise SpeechError(
                "Could not reach VoiceStudio at %s." % self.base_url
            )
        if response.status_code != 200:
            raise SpeechError(
                "VoiceStudio returned HTTP %s." % response.status_code
            )
        if not response.content:
            raise SpeechError("VoiceStudio returned empty audio.")
        return response.content


def _terminate_process(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            return
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


class Afplay(object):
    """Play one WAV with macOS afplay. stop() ends that process."""

    def __init__(self, popen=None):
        self._popen = popen or subprocess.Popen
        self._lock = threading.Lock()
        self._proc = None

    def play(self, wav_bytes, still_current):
        if not still_current():
            return False
        fd, path = tempfile.mkstemp(prefix="tinytalk-speech-", suffix=".wav")
        try:
            os.write(fd, wav_bytes)
            os.close(fd)
            fd = None
            if not still_current():
                return False
            proc = self._popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            with self._lock:
                if not still_current():
                    _terminate_process(proc)
                    return False
                self._proc = proc
            code = proc.wait()
            with self._lock:
                if self._proc is proc:
                    self._proc = None
            return code == 0 and still_current()
        finally:
            if fd is not None:
                os.close(fd)
            try:
                os.remove(path)
            except OSError:
                pass

    def stop(self):
        with self._lock:
            proc = self._proc
            self._proc = None
        _terminate_process(proc)


class SpeechController(object):
    """One current clip, and at most one stopped request still generating."""

    def __init__(self, client, player, output=None):
        self.client = client
        self.player = player
        self.output = output or print
        self._lock = threading.Lock()
        self._generation = 0
        self._phase = None
        self._thread = None
        self._threads = []

    def speak_last(self, messages):
        text = last_assistant_answer(messages)
        if text is None:
            self.output("There isn't an assistant answer to speak yet.")
            return
        with self._lock:
            self._threads = [thread for thread in self._threads if thread.is_alive()]
            if self._phase in ("starting", "generating", "playing"):
                self.output(
                    "Speech is already in progress. "
                    "TinyTalk did not start another clip."
                )
                return
            if len(self._threads) >= 2:
                self.output(
                    "A stopped request is still generating. "
                    "TinyTalk did not start another clip."
                )
                return
            self._generation += 1
            generation = self._generation
            self._phase = "starting"
            thread = threading.Thread(
                target=self._run,
                args=(text, generation),
            )
            thread.daemon = True
            self._thread = thread
            self._threads.append(thread)
            thread.start()
        self.output("Speaking the last answer…")

    def stop(self, quiet=False):
        with self._lock:
            phase = self._phase
            self._generation += 1
            if phase in ("starting", "generating", "playing"):
                self._phase = None
        self.player.stop()
        if phase not in ("starting", "generating", "playing"):
            if not quiet:
                self.output("Nothing is playing.")
            return
        self.output("Stopped playback.")
        if phase in ("starting", "generating"):
            self.output(
                "VoiceStudio may still finish generating that clip on the server. "
                "TinyTalk will not play it."
            )

    def join(self, timeout=None):
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout)

    def _current(self, generation):
        with self._lock:
            return generation == self._generation

    def _run(self, text, generation):
        with self._lock:
            if generation != self._generation:
                return
            self._phase = "generating"
        try:
            wav = self.client.synthesize(text)
        except SpeechError as exc:
            with self._lock:
                current = generation == self._generation
                if current:
                    self._phase = None
            if current:
                self.output("%s Text chat still works." % exc)
            return
        except Exception:
            with self._lock:
                current = generation == self._generation
                if current:
                    self._phase = None
            if current:
                self.output(
                    "VoiceStudio speech failed. Text chat still works."
                )
            return
        if not self._current(generation):
            return
        with self._lock:
            if generation != self._generation:
                return
            self._phase = "playing"
        played = False
        try:
            played = self.player.play(wav, lambda: self._current(generation))
        except Exception:
            played = False
            if self._current(generation):
                self.output(
                    "Could not play the audio. Text chat still works."
                )
        with self._lock:
            if generation == self._generation:
                self._phase = None
        if played and self._current(generation):
            self.output("Played the last answer.")
