"""Lets testing scripts import the real prompt_enhancer.py module outside
Forge. Stubs the Forge-specific imports (`modules.scripts`,
`modules.ui_components`) with minimal fakes so prompt_enhancer's
module-level code (config loading, globals, function definitions)
succeeds. The UI itself is only built when Forge calls `ui()`, which
nothing here does.

Usage from another script:
    from experiments._pe_bootstrap import pe
    sp = pe._assemble_system_prompt("Default")
    style = pe._build_style_string(mods)

The test harness MUST use this to test against the same functions
Forge runs. Mirroring them in the harness is what introduced drift
that hid real integration bugs.
"""

import importlib.util
import os
import sys
import types


_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _install_stubs() -> None:
    """Minimal Forge module stubs — just enough to let prompt_enhancer's
    module-level code + function definitions succeed."""
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


def _import_prompt_enhancer():
    """Load prompt_enhancer.py as a module named `pe`."""
    _install_stubs()
    path = os.path.join(_EXT_DIR, "scripts", "prompt_enhancer.py")
    spec = importlib.util.spec_from_file_location("pe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pe"] = module
    spec.loader.exec_module(module)
    return module


pe = _import_prompt_enhancer()
