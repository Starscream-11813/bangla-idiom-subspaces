"""The unified CLI (run_experiments.py) covers every experiment, group and option (CPU, seconds)."""
import importlib
import json
from pathlib import Path

import pytest

import run_experiments as run

REPO = Path(__file__).resolve().parents[2]


def test_every_experiment_has_a_module_with_run_and_a_known_group():
    assert sorted(run.ORDER, key=lambda c: int(c[1:])) == [f"X{k}" for k in range(1, 20)]
    for code, (module, folder, splits, title, group) in run.EXPERIMENTS.items():
        assert callable(importlib.import_module(f"experiments.{module}").run), code
        assert group in run.GROUPS and code in run.GROUPS[group]["order"]
        assert set(splits) <= {0, 1} and title


def test_every_group_has_its_gate_first_and_its_checks():
    for group, spec in run.GROUPS.items():
        assert spec["order"][0] == spec["gate"]
        assert callable(importlib.import_module(f"experiments.{spec['preflight']}").preflight)
        assert callable(importlib.import_module(f"experiments.{spec['verify']}").verify)


def test_every_group_option_is_a_parser_option():
    args = run.parser().parse_args(["list"])
    for group, spec in run.GROUPS.items():
        for key in spec["options"]:
            assert hasattr(args, key), (group, key)
        for key in spec["smoke"]:
            assert hasattr(args, key) or key == "mlp_small", (group, key)


def test_committed_results_have_one_folder_per_experiment_and_a_manifest():
    root = REPO / "results" / "experiments"
    for code, (_, folder, _, _, _) in run.EXPERIMENTS.items():
        if code == "X14":          # the optional decoder arm was not run
            assert not (root / folder).exists()
            continue
        manifest = json.loads((root / folder / "MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["experiment"] == folder


def test_verify_dispatches_check_codes_to_their_group(monkeypatch):
    calls = []

    def fake(group):
        def verify(results, out, only, *x7):
            calls.append((group, only))
            return 0
        return verify

    real = importlib.import_module
    monkeypatch.setattr(run.importlib, "import_module", lambda name: type("M", (), {
        "verify": staticmethod(fake(name.rsplit(".", 1)[1]))}) if name.split(".")[-1].startswith("verify")
        else real(name))
    with pytest.raises(SystemExit) as stop:
        run.main(["verify", "--check", "X5,X10,X16"])
    assert stop.value.code == 0
    assert calls == [("verify", ["X5"]), ("verify2", ["X10"]), ("verify3", ["X16"])]
