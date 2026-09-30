from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _find_project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Cannot locate WorldSolver project root")


ROOT = _find_project_root()
PLATFORM_ROOT = ROOT / "platform"
CONFIGS_ROOT = PLATFORM_ROOT / "configs"
PROMPTS_ROOT = PLATFORM_ROOT / "prompts"
TEMPLATE_ROOT = PLATFORM_ROOT / "template"
SCRIPTS_ROOT = PLATFORM_ROOT / "scripts"

TASKS_ROOT = ROOT / "task"

EXPERIMENTS_ROOT = ROOT / "experiments"
SCAFFOLDS_ROOT = ROOT / "scaffolds"
WORKSPACES_ROOT = EXPERIMENTS_ROOT / "workspaces"
RESULTS_ROOT = ROOT / "result"
TASK_DRAFTS_ROOT = EXPERIMENTS_ROOT / "task_drafts"

VENDOR_ROOT = ROOT / "vendor"
GENESIS_ROOT = VENDOR_ROOT / "genesis-world"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


# Keys models.local.yaml is allowed to override on an existing model entry.
CREDENTIAL_KEYS = frozenset({"base_url", "api_key"})


def load_models(configs_root: Path) -> dict[str, dict[str, Any]]:
    """Load model entries, merging the gitignored ``models.local.yaml``.

    ``models.yaml`` is the authoritative, committed source for every
    non-credential field (provider, executable, model, effort, ...).
    ``models.local.yaml`` is a credentials-only overlay: for each entry it may
    set ``base_url`` and ``api_key`` on a model that already exists in
    ``models.yaml``. Local values win; each returned entry is the merged result.
    """
    shared = load_yaml(configs_root / "models.yaml")
    models = {name: dict(entry) for name, entry in shared.get("models", {}).items()}
    local_path = configs_root / "models.local.yaml"
    if local_path.is_file():
        local = load_yaml(local_path)
        stray = set(local) - {"models"}
        if stray:
            raise ValueError(
                f"models.local.yaml carries credentials only (a 'models' mapping); "
                f"move {sorted(stray)} into models.yaml."
            )
        for name, creds in local.get("models", {}).items():
            if name not in models:
                raise ValueError(
                    f"models.local.yaml references '{name}', which is not defined in "
                    "models.yaml; add the provider/executable entry there (the local "
                    "file sets credentials only)."
                )
            unknown = set(creds) - CREDENTIAL_KEYS
            if unknown:
                raise ValueError(
                    f"models.local.yaml may only override {sorted(CREDENTIAL_KEYS)} "
                    f"for '{name}', got {sorted(unknown)}; move those fields to "
                    "models.yaml."
                )
            models[name].update(creds)
    return models


def sanitize_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    """Drop credentials from a model config for on-disk logging.

    ``load_models`` returns every merged entry including its ``api_key``; any
    file written to disk must exclude keys whose names contain ``key``. Mirrors
    the inline filter formerly in ``artifacts/manifest.py``.
    """
    return {
        key: value for key, value in model_config.items() if "key" not in key.lower()
    }


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


def ensure_within(path: Path, root: Path) -> Path:
    resolved, base = path.resolve(), root.resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"Path escapes root {base}: {resolved}")
    return resolved
