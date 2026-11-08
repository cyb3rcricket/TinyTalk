"""Local Gradio UI. One turn goes through tinytalk.handle_user_line."""

import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import gradio as gr
from provider import ConfigError, build_provider, load_settings
from tinytalk import (
    SessionSources,
    handle_user_line,
    load_dotenv,
    load_soul,
    open_memory,
)

_SEPARATOR = "-" * 30


class NoOpSpeech(object):
    """The web UI does not play audio. /speak and /stop stay harmless."""

    def speak_last(self, messages):
        return None

    def stop(self, *args, **kwargs):
        return None


def _call_quietly(function, *args):
    """Run function with debug off and return (result, captured stdout).

    Matches the stdout capture in mcp_server and scripts/dream_pass.py.
    The captured text is kept for the chat bubble instead of sent to stderr.
    """
    previous = os.environ.get("TINYTALK_DEBUG")
    os.environ["TINYTALK_DEBUG"] = ""
    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            result = function(*args)
    finally:
        if previous is None:
            os.environ.pop("TINYTALK_DEBUG", None)
        else:
            os.environ["TINYTALK_DEBUG"] = previous
    return result, captured.getvalue()


def _visible_messages(messages):
    visible = []
    for message in messages or []:
        role = (message or {}).get("role")
        if role not in ("user", "assistant"):
            continue
        visible.append({"role": role, "content": message.get("content") or ""})
    return visible


def _appended_assistant(before, after):
    """The assistant message this call added, or None for commands and errors."""
    if after == before:
        return None
    if not after or after[-1].get("role") != "assistant":
        return None
    if len(after) > len(before) and after[:len(before)] == before:
        return after[-1]
    if after[-1] != (before[-1] if before else None):
        return after[-1]
    if len(after) == len(before):
        return after[-1]
    return None


def _other_lines(captured, reply, label):
    """Stdout lines that are not the printed reply or the separator."""
    lines = captured.splitlines()
    printed = ("%s: %s" % (label, reply)).splitlines()
    kept = []
    index = 0
    skipped = False
    while index < len(lines):
        if (
            not skipped
            and printed
            and lines[index:index + len(printed)] == printed
        ):
            index += len(printed)
            skipped = True
            continue
        if lines[index] == _SEPARATOR:
            index += 1
            continue
        kept.append(lines[index])
        index += 1
    while kept and kept[0] == "":
        del kept[0]
    while kept and kept[-1] == "":
        del kept[-1]
    return kept


def _display_reply(captured, reply, label):
    extras = _other_lines(captured, reply, label)
    if not extras:
        return reply
    return reply + "\n\n" + "\n".join(extras)


def create_app(provider=None, memory=None, soul=None, speech=None, sources=None):
    """Build a Blocks app and do not launch it.

    Any argument left as None is built the way tinytalk.main builds it,
    except speech, which is a no-op so the page never opens an audio player.
    """
    if provider is None or memory is None or soul is None:
        load_dotenv(Path(__file__).with_name(".env"))
    if provider is None:
        provider = build_provider(load_settings(os.environ))
    if memory is None:
        memory = open_memory()
    if soul is None:
        soul = load_soul()
    if speech is None:
        speech = NoOpSpeech()
    if sources is None:
        sources = SessionSources()

    def submit_message(text, chat, messages, sources_state):
        if sources_state is None:
            sources_state = SessionSources()
        if not (text or "").strip():
            # Leave the visible transcript alone, including command bubbles
            # that are not part of the model history.
            if chat is None:
                chat = _visible_messages(messages)
            return chat, [] if messages is None else messages, sources_state, ""
        if messages is None:
            messages = []
        else:
            messages = list(messages)
        before = list(messages)
        updated, captured = _call_quietly(
            handle_user_line,
            text,
            messages,
            provider,
            memory,
            soul,
            speech,
            sources_state,
        )
        chat = _visible_messages(updated)
        assistant = _appended_assistant(before, updated)
        if assistant is not None:
            bubble = _display_reply(
                captured, assistant.get("content") or "", provider.label
            )
            for index in range(len(chat) - 1, -1, -1):
                if chat[index]["role"] == "assistant":
                    chat[index] = {"role": "assistant", "content": bubble}
                    break
        else:
            output = captured.strip("\n")
            if output:
                chat.append({"role": "user", "content": text.strip()})
                chat.append({"role": "assistant", "content": output})
        return chat, updated, sources_state, ""

    def clear_conversation():
        return [], [], SessionSources(), ""

    with gr.Blocks(title="TinyTalk", analytics_enabled=False) as demo:
        gr.Markdown("# TinyTalk")
        chatbot = gr.Chatbot(
            show_label=False,
            group_consecutive_messages=False,
            buttons=[],
        )
        textbox = gr.Textbox(label="Message", lines=1)
        clear = gr.Button("Clear")
        messages_state = gr.State([])
        sources_state = gr.State(sources)
        textbox.submit(
            submit_message,
            inputs=[textbox, chatbot, messages_state, sources_state],
            outputs=[chatbot, messages_state, sources_state, textbox],
        )
        clear.click(
            clear_conversation,
            outputs=[chatbot, messages_state, sources_state, textbox],
        )
    return demo


def main():
    try:
        app = create_app()
    except ConfigError as exc:
        print("Configuration error: %s" % exc)
        return 1
    port = int(os.environ.get("TINYTALK_WEB_PORT", "7861"))
    app.launch(share=False, server_name="127.0.0.1", server_port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
