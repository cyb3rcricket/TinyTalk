"""Keep memory tests away from any real MemPalace.

Import this before tinytalk or mempalace. It points HOME, the XDG
directories, and MemPalace's config and palace paths at one temporary root.
open_isolated_memory refuses to open a store unless every path it would
write resolves inside that root.

Set TINYTALK_TEST_ROOT to a directory under the system temp directory to
reuse it between runs (MemPalace keeps its embedding model under HOME, so a
fresh root downloads the model again). Anything else is refused.
"""

import atexit
import os
import shutil
import sys
import tempfile

if "mempalace" in sys.modules:
    raise RuntimeError("Import memory_isolation before mempalace so its paths are isolated.")

_SYSTEM_TEMP = os.path.realpath(tempfile.gettempdir())
_given = os.environ.get("TINYTALK_TEST_ROOT", "").strip()
if _given:
    ROOT = os.path.realpath(_given)
    if not ROOT.startswith(_SYSTEM_TEMP + os.sep):
        raise RuntimeError("TINYTALK_TEST_ROOT must be inside %s." % _SYSTEM_TEMP)
    os.makedirs(ROOT, exist_ok=True)
else:
    ROOT = os.path.realpath(tempfile.mkdtemp(prefix="tinytalk-test-"))
    atexit.register(shutil.rmtree, ROOT, True)

_HOME = os.path.join(ROOT, "home")
os.environ.update({
    "HOME": _HOME,
    "XDG_CONFIG_HOME": os.path.join(_HOME, ".config"),
    "XDG_DATA_HOME": os.path.join(_HOME, ".local", "share"),
    "XDG_CACHE_HOME": os.path.join(_HOME, ".cache"),
    "MEMPALACE_CONFIG_DIR": os.path.join(_HOME, ".config", "mempalace"),
    "MEMPALACE_PALACE_PATH": os.path.join(ROOT, "default-palace"),
})
os.environ.pop("MEMPAL_PALACE_PATH", None)
os.makedirs(_HOME, exist_ok=True)


def inside_root(path):
    return os.path.realpath(path).startswith(ROOT + os.sep)


def require_inside(path, what):
    if not path or not inside_root(path):
        raise RuntimeError("Refusing to use %s outside the test root: %r" % (what, path))


def check_default_paths():
    """MemPalace's own defaults must also land in the test root."""
    from mempalace.config import MempalaceConfig
    from mempalace.knowledge_graph import DEFAULT_KG_PATH

    config = MempalaceConfig()
    require_inside(str(config.config_dir), "the MemPalace config directory")
    require_inside(config.palace_path, "the default palace")
    require_inside(DEFAULT_KG_PATH, "the default knowledge graph")


def open_isolated_memory(temp_dir):
    """Open TinyTalk memory in temp_dir after checking every path it can write."""
    import tinytalk
    from mempalace.config import MempalaceConfig

    require_inside(temp_dir, "the temporary store")
    check_default_paths()
    palace_path = os.path.join(temp_dir, "palace")
    kg_path = os.path.join(temp_dir, "knowledge_graph.sqlite3")
    require_inside(MempalaceConfig(palace_path=palace_path).palace_path, "the palace")
    require_inside(kg_path, "the knowledge graph")

    memory = tinytalk.open_memory(palace_path=palace_path, kg_path=kg_path)
    if memory.palace is None or memory.kg is None or memory.add_drawer is None:
        raise RuntimeError("MemPalace did not open in the test root.")
    require_inside(memory.palace_path, "the opened palace")
    require_inside(memory.kg.db_path, "the opened knowledge graph")
    mcp = sys.modules["mempalace.mcp_server"]
    require_inside(mcp._config.palace_path, "the MemPalace drawer tools")
    return memory


def temp_dir(prefix):
    """A TemporaryDirectory inside the test root."""
    return tempfile.TemporaryDirectory(prefix=prefix, dir=ROOT)
