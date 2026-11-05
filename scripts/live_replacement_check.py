"""Re-check replacement against the real default palace and configured model.

Run from the repository root: python scripts/live_replacement_check.py
This writes Checkship Alpha, then Checkship Beta, to test_spaceship_name.
Beta stays saved; the script does not restore the old value or edit README.
Exit codes: 0 = all checks passed, 1 = check/runtime failure, 2 = no provider.
"""

import io
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREDICATE = "test_spaceship_name"
ALPHA = "Checkship Alpha"
BETA = "Checkship Beta"
QUESTION = "What was the previous name of my test spaceship?"
CHECKS = (
    "Alpha was saved as current",
    "Beta replaced Alpha",
    "Current test_spaceship_name is Checkship Beta",
    "Checkship Alpha is recoverable from facts-history through chat evidence",
    "Checkship Alpha is not current",
)


def quiet_call(function, *args):
    """Keep library/debug output and raw provider errors out of the report."""
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return function(*args)


def profile_entry(core, memory):
    return next(
        entry for entry in quiet_call(core.read_current_profile, memory)
        if entry["predicate"] == PREDICATE
    )


def main():
    if os.environ.get("TINYTALK_TEST_ROOT") or "memory_isolation" in sys.modules:
        print("FAIL: refusing a test-isolation environment; use your normal TinyTalk session.")
        print("FAIL: live replacement check")
        return 1

    def build():
        from provider import build_provider, load_settings
        from tinytalk import load_dotenv

        load_dotenv(ROOT / ".env")
        return build_provider(load_settings(os.environ))

    try:
        backend = quiet_call(build)
        if backend is None or not callable(getattr(backend, "complete", None)):
            raise RuntimeError("No usable provider was built")
    except Exception as exc:
        print(
            "Provider could not be built (%s). Check the normal TinyTalk dependencies "
            "and provider settings. No memory was opened." % type(exc).__name__
        )
        print("FAIL: live replacement check")
        return 2

    secrets = sorted({
        value for name, value in os.environ.items()
        if value and any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    }, key=len, reverse=True)

    def safe(text):
        for secret in secrets:
            text = text.replace(secret, "[redacted]")
        return text

    memory = None
    outcomes = [None] * len(CHECKS)
    errors = []
    stage = "opening default memory"
    print("This writes to your default palace and leaves test_spaceship_name as Checkship Beta.")
    try:
        import memory_core as core

        memory = quiet_call(core.open_memory)
        if (memory is None or memory.palace is None or memory.kg is None
                or not getattr(memory, "provenance", True)
                or not all(callable(getattr(memory, name, None)) for name in (
                    "add_drawer", "list_drawers", "update_drawer", "get_drawer",
                ))):
            raise RuntimeError("Default memory lacks the required drawer/graph support")

        stage = "saving Checkship Alpha"
        print("Saving Checkship Alpha...")
        alpha = quiet_call(core.remember_fact, "Remember this: my test spaceship is named " + ALPHA,
                           memory, backend)
        print(safe("Alpha status: %s; %s" % (alpha["status"], alpha["message"])))
        alpha_profile = profile_entry(core, memory)
        outcomes[0] = (
            alpha["status"] in ("saved", "replaced", "already_saved")
            and alpha_profile["state"] == "current" and alpha_profile["value"] == ALPHA
        )

        # MemPalace's replacement boundaries have second precision. Keep the
        # two changes distinct so the history path can establish their order.
        time.sleep(1.1)
        stage = "saving Checkship Beta"
        print("Saving Checkship Beta...")
        beta = quiet_call(core.remember_fact, "Remember this: my test spaceship is named " + BETA,
                          memory, backend)
        print(safe("Beta status: %s; %s" % (beta["status"], beta["message"])))
        outcomes[1] = bool(outcomes[0] and beta["status"] == "replaced")

        stage = "reading the current profile"
        current = profile_entry(core, memory)
        outcomes[2] = current["state"] == "current" and current["value"] == BETA
        outcomes[4] = not (current["state"] == "current" and current["value"] == ALPHA)

        stage = "reading chat history evidence"
        message, sources = quiet_call(core.historical_context_message, QUESTION, memory)
        outcomes[3] = (
            "Previous name: " + ALPHA in message["content"].splitlines()
            and bool(alpha_profile.get("source_drawer_id"))
            and any(
                source["kind"] == "archived fact"
                and source.get("id") == alpha_profile["source_drawer_id"]
                and core.test_spaceship_fact_name(source.get("text")) == ALPHA
                for source in sources
            )
        )
    except Exception as exc:
        errors.append("%s failed (%s)" % (stage, type(exc).__name__))
    finally:
        for label, resource in (
            ("closing memory", getattr(memory, "kg", None)),
            ("closing provider", getattr(backend, "client", None)),
        ):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    quiet_call(close)
                except Exception as exc:
                    errors.append("%s failed (%s)" % (label, type(exc).__name__))

    for label, passed in zip(CHECKS, outcomes):
        print("%s: %s%s" % (
            "PASS" if passed else "FAIL", label, " (not completed)" if passed is None else "",
        ))
    for error in errors:
        print("ERROR: " + error)
    passed = all(outcomes) and not errors
    print(("PASS" if passed else "FAIL") + ": live replacement check")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
