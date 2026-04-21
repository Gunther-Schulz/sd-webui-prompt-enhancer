"""Lets testing scripts import the real prompt_enhancer.py module outside
Forge. Stubs the Forge-specific imports (`modules.scripts`,
`modules.ui_components`, `modules.shared`) with minimal fakes so
prompt_enhancer's module-level code (config loading, globals) succeeds,
even though the Gradio UI construction at class definition time will
attempt to build components with no tab context.

Usage from another script:
    from anima_tagger.scripts._pe_bootstrap import pe
    sp = pe._assemble_system_prompt("Detailed", None, 0)
    style = pe._build_style_string(mods, mode="prose")

The test harness MUST use this to test against the same functions
Forge runs. Mirroring them in the harness is what introduced drift
that hid real integration bugs.
"""

import importlib.util
import os
import sys
import types


_EXT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _install_stubs() -> None:
    """Minimal Forge module stubs — just enough to let prompt_enhancer's
    module-level code + function definitions succeed. UI construction
    will still fail; we don't need it for testing."""
    if "modules" in sys.modules:
        return

    # modules
    modules = types.ModuleType("modules")
    sys.modules["modules"] = modules

    # modules.scripts
    scripts = types.ModuleType("modules.scripts")
    class Script:
        pass
    scripts.Script = Script
    scripts.AlwaysVisible = "AlwaysVisible"
    scripts.ScriptFile = object
    modules.scripts = scripts
    sys.modules["modules.scripts"] = scripts

    # modules.ui_components
    ui = types.ModuleType("modules.ui_components")
    class ToolButton:
        def __init__(self, *a, **kw):
            pass
    ui.ToolButton = ToolButton
    modules.ui_components = ui
    sys.modules["modules.ui_components"] = ui

    # modules.shared (read-only opts dict)
    shared = types.ModuleType("modules.shared")
    class _Opts:
        data = {
            "anima_tagger_semantic_threshold": 0.7,
            "anima_tagger_semantic_min_post_count": 50,
            "anima_tagger_enable_reranker": True,
            "anima_tagger_enable_cooccurrence": True,
            "anima_tagger_query_expansion": True,
            "anima_tagger_device": "auto",
            "anima_tagger_compound_split": True,
            "anima_tagger_slot_fill": True,
        }
    shared.opts = _Opts()
    shared.cmd_opts = types.SimpleNamespace(disable_extra_extensions=False,
                                            disable_all_extensions=False)
    modules.shared = shared
    sys.modules["modules.shared"] = shared


def _import_prompt_enhancer():
    """Load prompt_enhancer.py as a module named `pe`. Catches the
    UI class-construction failure so only module-level defs survive
    (which is all we need for testing)."""
    _install_stubs()
    path = os.path.join(_EXT_DIR, "scripts", "prompt_enhancer.py")
    spec = importlib.util.spec_from_file_location("pe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pe"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # UI construction fails without a real Gradio context, but the
        # module-level functions and globals are already populated by
        # the time we get here. Keep going.
        print(f"[pe_bootstrap] non-fatal: {type(e).__name__}: {e}")
    return module


pe = _import_prompt_enhancer()
