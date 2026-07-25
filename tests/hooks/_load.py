"""Load a script under ``scripts/hooks/`` as an importable module for tests."""

import importlib.util
from pathlib import Path
from types import ModuleType

_HOOKS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hooks"


def load_hook(name: str) -> ModuleType:
    """Import ``scripts/hooks/<name>.py`` and return the loaded module."""
    path = _HOOKS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
