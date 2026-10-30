"""Observable progress files stay readable and distinguish terminal outcomes."""
import json
from pathlib import Path
import threading
import time

import pytest

from idiom_subspace.progress import Progress, atomic_json, progress_iter


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate(), "Timed out waiting for a progress heartbeat"


def test_atomic_progress_is_readable_during_updates(tmp_path):
    path = tmp_path / "worker" / "progress.json"
    stop, errors, observations = threading.Event(), [], []
    with Progress(path, heartbeat_seconds=.05, model="unit-model") as reporter:
        def read_while_running():
            while not stop.is_set():
                try:
                    value = _read(path)
                    assert value["status"] == "running"
                    assert isinstance(value["context"], dict)
                    observations.append(value["completed"])
                except PermissionError:
                    # Windows may deny opening during the instantaneous swap;
                    # the dashboard reader retries/falls back on an OSError.
                    pass
                except Exception as error:
                    errors.append(error)
                stop.wait(.002)
        reader = threading.Thread(target=read_while_running)
        reader.start()
        try:
            for completed in range(20):
                reporter.update(stage="extract", completed=completed, total=20)
        finally:
            stop.set()
            reader.join(timeout=2)
    assert not errors
    assert observations
    assert _read(path)["status"] == "completed"
    assert not list(path.parent.glob("*.tmp"))


def test_heartbeat_preserves_work_counts_and_stops_after_close(tmp_path):
    path = tmp_path / "progress.json"
    reporter = Progress(path, heartbeat_seconds=.05)
    try:
        reporter.update(stage="extract", completed=2, total=10, model="example/model")
        before = _read(path)
        _until(lambda: _read(path)["updated_at"] != before["updated_at"])
        after = _read(path)
        assert after["completed"] == 2 and after["fraction"] == .2
        assert after["context"]["model"] == "example/model"
        assert after["elapsed_seconds"] >= before["elapsed_seconds"]
        reporter.close("interrupted", "operator stopped this worker")
        final = path.read_bytes()
        time.sleep(.12)
        assert path.read_bytes() == final
        assert _read(path)["status"] == "interrupted"
    finally:
        reporter.close("interrupted")


def test_stage_history_records_partial_work_and_fraction_is_stage_only(tmp_path):
    path = tmp_path / "progress.json"
    with Progress(path) as reporter:
        reporter.update(stage="extract", completed=3, total=10)
        assert _read(path)["unit"] == "items"
        reporter.update(stage="analysis")
        stage = _read(path)
        previous = [row for row in stage["history"] if row["stage"] == "extract"][0]
        assert previous["completed"] == 3 and previous["total"] == 10
        assert stage["completed"] == 0 and stage["total"] is None and stage["fraction"] is None
        assert stage["eta_seconds"] is None
        assert stage["unit"] == "tasks"
        reporter.update(completed=15, total=10)
        assert _read(path)["fraction"] == 1
        reporter.update(completed=-2)
        assert _read(path)["fraction"] == 0


@pytest.mark.parametrize("exception,status", [(KeyboardInterrupt("cancelled"), "interrupted"),
                                               (ValueError("bad configuration"), "failed")])
def test_exception_exit_does_not_report_success(tmp_path, exception, status):
    path = tmp_path / "progress.json"
    with pytest.raises(type(exception)):
        with Progress(path) as reporter:
            reporter.update(stage="analysis", completed=1, total=4)
            raise exception
    value = _read(path)
    assert value["status"] == status
    assert value["completed"] == 1 and value["fraction"] == .25
    assert value["message"] == str(exception)


def test_atomic_json_rejects_nonfinite_values_without_replacing_previous_file(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"status": "running"})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(path, {"fraction": float("nan")})
    assert path.read_bytes() == before
    assert list(progress_iter([1, 2, 3], enabled=False)) == [1, 2, 3]


def test_stage_eta_and_layer_milestones_are_labelled_estimates(tmp_path):
    path = tmp_path / "progress.json"
    with Progress(path, rate_window_seconds=0) as reporter:
        reporter.update(stage="grid", completed=0, total=100)
        time.sleep(.05)
        reporter.update(completed=50)
        state = _read(path)
        assert state["eta_seconds"] is not None and state["eta_seconds"] > 0
        assert state["eta_kind"] == "stage_throughput_estimate" and state["rate_per_second"] > 0
        reporter.start_milestone("layer", total=4)
        time.sleep(.02)
        reporter.milestone("layer", 1, 4)
        layer = _read(path)["milestones"]["layer"]
        assert layer["done"] == 1 and layer["total"] == 4
        assert layer["mean_seconds"] > 0 and layer["eta_seconds"] >= 0 and len(layer["durations"]) == 1
        reporter.update(completed=100)
        assert _read(path)["eta_seconds"] is None


def test_echo_prints_readable_lines_when_output_is_redirected(tmp_path):
    import io
    stream = io.StringIO()
    with Progress(tmp_path / "progress.json", echo=True, echo_interval=0, stream=stream,
                  model="xlm-roberta-large", seed=0, label_seed=1) as reporter:
        reporter.start_milestone("layer", total=3)
        reporter.update(stage="grid", completed=5, total=10, layer=12)
        reporter.milestone("layer", 1, 3)
    text = stream.getvalue()
    assert "\r" not in text
    assert "xlm-roberta-large split0 tax1" in text
    assert "grid 5/10" in text and "layer 1/3" in text and "(L12)" in text


def test_disabled_reporter_has_no_journal_or_heartbeat(tmp_path):
    path = tmp_path / "disabled" / "progress.json"
    with Progress(path, enabled=False, heartbeat_seconds=.05) as reporter:
        reporter.update(stage="extract", completed=3, total=10)
        assert reporter._thread is None
    assert not path.parent.exists()
