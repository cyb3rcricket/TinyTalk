"""Model providers for TinyTalk.

Ollama stays on the local chat-completions endpoint. Grok uses the xAI
Responses API. Both accept the same list of {role, content} messages and
return plain assistant text.

Grok calls are stateless: TinyTalk resends its own trimmed history and sets
store=false. previous_response_id is not used, so xAI does not keep a
conversation that could outlive MAX_TURNS. The request is plain text
generation: tools, tool_choice, and search_parameters are omitted. The SDK's
own retries are disabled so TinyTalk's three attempts are the only ones.
"""

import time

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
XAI_BASE_URL = "https://api.x.ai/v1"
# Verified against the xAI docs (Grok 4.7 model page, September 2026).
# This is an API model ID, not a Grok Build product name or reasoning effort.
DOCUMENTED_XAI_MODEL = "grok-4.7"

REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (0.5, 1.0)
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class ConfigError(Exception):
    """The selected provider is missing a setting required to start."""


class ProviderError(Exception):
    """A model call failed or returned nothing usable.

    The chat loop must not record the turn when this is raised.
    """


class Settings(object):
    def __init__(
        self,
        provider,
        ollama_model,
        ollama_base_url,
        xai_api_key,
        xai_model,
    ):
        self.provider = provider
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.xai_api_key = xai_api_key
        self.xai_model = xai_model


def load_settings(env):
    """Read provider settings from a mapping. Does not create a client."""
    raw_provider = env.get("TINYTALK_PROVIDER")
    provider = (raw_provider or DEFAULT_PROVIDER).strip().lower()
    if provider not in ("ollama", "grok"):
        raise ConfigError(
            "TINYTALK_PROVIDER must be 'ollama' or 'grok' "
            "(got %r). TinyTalk did not start." % (raw_provider,)
        )

    ollama_model = (env.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
    if not ollama_model:
        ollama_model = DEFAULT_OLLAMA_MODEL
    ollama_base_url = (env.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL).strip()
    if not ollama_base_url:
        ollama_base_url = DEFAULT_OLLAMA_BASE_URL

    xai_api_key = (env.get("XAI_API_KEY") or "").strip()
    xai_model = (env.get("XAI_MODEL") or "").strip()
    if provider == "grok":
        missing = []
        if not xai_api_key:
            missing.append("XAI_API_KEY")
        if not xai_model:
            missing.append("XAI_MODEL")
        if missing:
            raise ConfigError(
                "Grok mode needs %s. Put them in the environment or .env. "
                "A verified model ID is %s. TinyTalk did not fall back to Ollama."
                % (" and ".join(missing), DOCUMENTED_XAI_MODEL)
            )

    return Settings(
        provider=provider,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url.rstrip("/"),
        xai_api_key=xai_api_key,
        xai_model=xai_model,
    )


def _field(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def text_from_chat_completion(response):
    """Plain text from an Ollama / chat-completions response."""
    choices = _field(response, "choices") or []
    if not choices:
        return ""
    message = _field(choices[0], "message")
    content = _field(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            else:
                text = _field(part, "text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def text_from_responses(response):
    """Plain assistant text from a Responses API payload.

    Reasoning items are ignored. A non-completed status is a failure, not text.
    """
    status = _field(response, "status")
    if status not in (None, "completed"):
        raise ProviderError(
            "xAI returned status %r instead of a completed answer. "
            "TinyTalk did not switch to another provider." % (status,)
        )
    error = _field(response, "error")
    if error:
        raise ProviderError(
            "xAI returned an error instead of an answer. "
            "TinyTalk did not switch to another provider."
        )

    direct = _field(response, "output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    parts = []
    for item in _field(response, "output") or []:
        item_type = _field(item, "type")
        if item_type not in (None, "message"):
            continue
        content = _field(item, "content")
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            block_type = _field(block, "type")
            text = _field(block, "text")
            if block_type in (None, "output_text") and isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def is_transient(exc):
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    if status in TRANSIENT_STATUS_CODES or (isinstance(status, int) and status >= 500):
        return True
    return False


def redact(text, secret):
    if not text:
        return ""
    if secret:
        text = text.replace(secret, "[redacted]")
    return " ".join(text.split())[:300]


def explain_failure(exc, service, model, secret):
    """User-facing text for an API failure. Never includes the API key."""
    status = getattr(exc, "status_code", None)
    stayed = "TinyTalk did not switch to another provider."
    if status == 401:
        if service == "xAI":
            return (
                "xAI rejected the API key (HTTP 401). Check XAI_API_KEY from "
                "the xAI console (https://console.x.ai/team/default/api-keys). " + stayed
            )
        return "The model server rejected the request (HTTP 401). " + stayed
    if status == 403:
        return (
            "%s refused access (HTTP 403). The key or team may not be allowed "
            "to use model %s. %s" % (service, model, stayed)
        )
    if status == 404:
        if service == "xAI":
            return (
                "xAI could not find model %r, or this team cannot use it (HTTP 404). "
                "Set XAI_MODEL to an ID your team can access. A verified ID is %s. %s"
                % (model, DOCUMENTED_XAI_MODEL, stayed)
            )
        return (
            "Ollama could not find model %r (HTTP 404). "
            "Pull it with: ollama pull %s. %s" % (model, model, stayed)
        )
    if status == 429:
        return (
            "%s rate limit reached (HTTP 429) after %d attempts. "
            "Wait and try again. %s" % (service, MAX_ATTEMPTS, stayed)
        )
    if isinstance(exc, APITimeoutError) or status == 408:
        return (
            "The request to %s timed out (limit %ss). %s"
            % (service, int(REQUEST_TIMEOUT.read), stayed)
        )
    if isinstance(exc, APIConnectionError):
        if service == "Ollama":
            return (
                "Could not reach Ollama. Start it and confirm it is serving "
                "the OpenAI-compatible API. %s" % stayed
            )
        return (
            "Could not reach the xAI API. Check the network connection and "
            "https://api.x.ai. %s" % stayed
        )
    if isinstance(status, int) and status >= 500:
        return (
            "%s had a server error (HTTP %s) after %d attempts. %s"
            % (service, status, MAX_ATTEMPTS, stayed)
        )
    if isinstance(exc, APIStatusError):
        return (
            "%s request failed (HTTP %s). %s %s"
            % (service, status, redact(str(exc), secret), stayed)
        )
    return "%s request failed. %s %s" % (service, redact(str(exc), secret), stayed)


def call_with_retries(operation, explain, sleep):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return operation()
        except ProviderError:
            raise
        except Exception as exc:
            if attempt + 1 >= MAX_ATTEMPTS or not is_transient(exc):
                raise ProviderError(explain(exc)) from exc
            delay = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
            sleep(delay)


class OllamaProvider(object):
    name = "ollama"

    def __init__(self, client, model, sleep=time.sleep):
        self.client = client
        self.model = model
        self.label = model
        self.sleep = sleep

    def complete(self, messages):
        def operation():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                timeout=REQUEST_TIMEOUT,
            )
            text = text_from_chat_completion(response)
            if not isinstance(text, str) or not text.strip():
                raise ProviderError(
                    "Ollama returned an empty response. "
                    "TinyTalk did not switch to another provider."
                )
            return text.strip()

        return call_with_retries(
            operation,
            lambda exc: explain_failure(exc, "Ollama", self.model, secret=""),
            self.sleep,
        )


class GrokProvider(object):
    name = "grok"
    label = "Grok"

    def __init__(self, client, model, api_key, sleep=time.sleep):
        self.client = client
        self.model = model
        self.api_key = api_key
        self.sleep = sleep

    def complete(self, messages):
        def operation():
            # store=false keeps the transcript on this machine. The same
            # message list is sent again next turn, already trimmed by TinyTalk.
            # Omit tools and tool_choice: the API returns HTTP 400 when
            # tool_choice is set without tools. Omit search_parameters:
            # that live-search field returns HTTP 410.
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                store=False,
                timeout=REQUEST_TIMEOUT,
            )
            text = text_from_responses(response)
            if not isinstance(text, str) or not text.strip():
                raise ProviderError(
                    "xAI returned an empty response. "
                    "TinyTalk did not switch to another provider."
                )
            return text.strip()

        return call_with_retries(
            operation,
            lambda exc: explain_failure(exc, "xAI", self.model, secret=self.api_key),
            self.sleep,
        )


def build_provider(settings, client_factory=OpenAI, sleep=time.sleep, http_client=None):
    """Create only the selected provider. Ollama mode never builds a Grok client.

    max_retries=0 leaves retry counting to call_with_retries (three attempts).
    http_client is for tests that supply a mocked transport.
    """
    client_kwargs = {
        "timeout": REQUEST_TIMEOUT,
        "max_retries": 0,
    }
    if http_client is not None:
        client_kwargs["http_client"] = http_client

    if settings.provider == "ollama":
        client_kwargs["base_url"] = settings.ollama_base_url
        client_kwargs["api_key"] = "ollama"
        client = client_factory(**client_kwargs)
        return OllamaProvider(client, settings.ollama_model, sleep=sleep)

    if settings.provider != "grok":
        raise ConfigError(
            "TINYTALK_PROVIDER must be 'ollama' or 'grok' (got %r)."
            % (settings.provider,)
        )
    if not settings.xai_api_key or not settings.xai_model:
        raise ConfigError(
            "Grok mode needs XAI_API_KEY and XAI_MODEL. "
            "TinyTalk did not fall back to Ollama."
        )
    client_kwargs["base_url"] = XAI_BASE_URL
    client_kwargs["api_key"] = settings.xai_api_key
    client = client_factory(**client_kwargs)
    return GrokProvider(
        client,
        settings.xai_model,
        api_key=settings.xai_api_key,
        sleep=sleep,
    )
