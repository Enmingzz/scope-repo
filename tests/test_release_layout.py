import hashlib
import json
from pathlib import Path
import tomllib

import scopd
import scopd_eval
import train_scopd


ROOT = Path(__file__).resolve().parents[1]


def test_public_modules_and_methods():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["tool"]["setuptools"]["py-modules"] == ["train_scopd"]
    for module in (scopd, scopd_eval, train_scopd):
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
    method = next(action for action in train_scopd.parser()._actions if action.dest == "method")
    assert method.choices == ("scopd", "scopd+")


def test_exported_sources_match_manifest():
    rows = json.loads((ROOT / "runtime/export_manifest.json").read_text())
    assert len({row["file"] for row in rows}) == len(rows)
    for row in rows:
        path = ROOT / row["file"]
        assert path.is_file(), row["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["export_sha256"], row["file"]


def test_runtime_overlays_match_manifest():
    manifest = json.loads((ROOT / "runtime/manifest.json").read_text())
    for mode, files in manifest["overlays"].items():
        for relative, expected in files.items():
            path = ROOT / "runtime/overlays" / mode / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, str(path)
