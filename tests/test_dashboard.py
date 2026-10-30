"""Exercise the read-only API over real ephemeral loopback HTTP connections."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import http.client
import json
from pathlib import Path
import threading
import tracemalloc
import pytest
from urllib.parse import quote

from idiom_subspace.dashboard_server import DashboardData, make_server


def test_dashboard_cannot_bind_over_an_existing_listener(tmp_path):
    first=make_server(tmp_path,port=0)
    try:
        with pytest.raises(OSError):make_server(tmp_path,port=first.server_port)
    finally:first.server_close()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")


def _progress(**overrides):
    value = dict(status="running", stage="extract", completed=2, total=8, fraction=.25,
                 updated_at=datetime.now(timezone.utc).isoformat(), context={"model": "example/model"}, history=[])
    value.update(overrides)
    return value


@contextmanager
def _serve(root):
    server = make_server(root, port=0)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(port, path, method="GET"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        body = response.read()
        headers = dict(response.getheaders())
        if "application/json" in headers.get("Content-Type", ""):
            def nonfinite(value):
                raise AssertionError(f"Nonstandard JSON number leaked through API: {value}")
            body = json.loads(body.decode("utf8"), parse_constant=nonfinite)
        return response.status, body, headers
    finally:
        connection.close()


def _snapshot(root):
    return {str(path.relative_to(root)): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in root.rglob("*") if path.is_file()}


def test_http_overview_detail_metrics_and_nonfinite_values_are_read_only(tmp_path):
    root = tmp_path / "outputs"
    folder = root / "split_0" / "labels_0" / "example_model"
    _json(folder / "progress.json", _progress(stage="analysis", fraction=.5, completed=1, total=2,
          context={"model": "example/model", "diagnostic": float("nan")}))
    _json(folder.parent / "split_audit.json", {"counts": {"space": 10, "probe": 4, "validation": 2, "test": 4}})
    _json(folder.parent / "label_audit.json", {"backend": "tfidf", "effective_k": 3, "literal_english_available": 0})
    (folder / "grid_layer_1.csv").write_text(
        "group,concept,method,M,source_condition,condition,R,L,R0,validation_R\n"
        "all,F,MLR,1,bn-fig,bn-fig,0.75,0.35,0.5,NaN\n"
        "cultural,F,MLR,1,bn-fig,bn-fig,0.8,0.3,0.5,0.7\n", encoding="utf8")
    (folder / "ssi_all_layers.csv").write_text("group,layer,SSI\ncultural,0,0.9\nall,1,NaN\nall,2,Infinity\n", encoding="utf8")
    (folder / "S3_inference.csv").write_text("group,SSI\nrural,0.9\nall,0.4\n", encoding="utf8")
    (folder / "pooling.csv").write_text("item_id,valid\na,True\nb,False\nc,True\n", encoding="utf8")
    (root / "private-artifact.txt").write_text("private contents", encoding="utf8")
    before = _snapshot(root)
    with _serve(root) as port:
        status, overview, headers = _request(port, "/api/overview")
        assert status == 200 and overview["total"] == 1 and overview["running"] == 1
        assert headers["Cache-Control"] == "no-store"
        job = overview["jobs"][0]
        assert "folder" not in job and job["context"]["diagnostic"] is None
        assert job["completed_layers"] == [1]
        assert job["pooling"] == {"items": 3, "valid": 2, "invalid": 1}
        status, detail, _ = _request(port, "/api/job?id=" + quote(job["id"]))
        assert status == 200 and len(detail["grid"]) == 1
        assert detail["grid"][0]["R"] == .75 and detail["grid"][0]["validation_R"] is None
        assert all(row["SSI"] is None for row in detail["ssi"])
        assert len(detail["ssi"]) == 2 and detail["inference"] == [{"group": "all", "SSI": .4}]
        assert detail["split"]["counts"]["test"] == 4
        assert detail["labels"]["backend"] == "tfidf"
        assert "folder" not in detail["job"]
    assert _snapshot(root) == before


def test_unknown_ids_traversal_and_artifact_files_are_not_served(tmp_path):
    root = tmp_path / "outputs"
    folder = root / "worker"
    _json(folder / "status.json", {"status": "completed", "model": "legacy-model"})
    (folder / "secret.csv").write_text("sensitive,artifact\n", encoding="utf8")
    paths = ["/api/job?id=unknown", "/api/job", "/api/job?id=../worker", "/worker/secret.csv",
             "/../worker/secret.csv", "/%2e%2e/worker/secret.csv", "/api/file?path=worker/secret.csv"]
    with _serve(root) as port:
        for path in paths:
            status, body, _ = _request(port, path)
            assert status == 404 and "error" in body
            assert "sensitive" not in json.dumps(body)
        status, _, _ = _request(port, "/api/overview", method="POST")
        assert status in {405, 501}


def test_stale_missing_and_legacy_heartbeats_are_not_reported_as_live(tmp_path):
    now = datetime.now(timezone.utc)
    _json(tmp_path / "fresh" / "progress.json", _progress())
    _json(tmp_path / "stale" / "progress.json", _progress(updated_at=(now-timedelta(minutes=2)).isoformat()))
    _json(tmp_path / "interrupted" / "progress.json", _progress(status="interrupted", updated_at=(now-timedelta(minutes=2)).isoformat()))
    _json(tmp_path / "old_running" / "status.json", {"status": "running", "model": "older-run"})
    _json(tmp_path / "old_completed" / "status.json", {"status": "completed", "model": "older-complete"})
    _json(tmp_path / "invalid_stamp" / "progress.json", _progress(updated_at="not-a-time"))
    _json(tmp_path / "naive_stamp" / "progress.json", _progress(updated_at=now.replace(tzinfo=None).isoformat()))
    with _serve(tmp_path) as port:
        status, overview, _ = _request(port, "/api/overview")
        assert status == 200 and overview["running"] == 2
        assert overview["completed"] == 1 and overview["failed"] == 1
        jobs = {job["relative_path"]: job for job in overview["jobs"]}
        assert jobs["stale"]["stale"] and not jobs["stale"]["live"]
        assert not jobs["interrupted"]["stale"]
        legacy = jobs["old_running"]
        assert legacy["stage"] == "legacy_run" and legacy["status"] == "running"
        assert legacy["heartbeat_unknown"] and not legacy["live"] and not legacy["telemetry_available"]
        assert "No live telemetry" in legacy["message"]
        assert jobs["invalid_stamp"]["heartbeat_unknown"]
        assert jobs["naive_stamp"]["live"]


def test_pending_work_has_no_fabricated_analysis_or_results(tmp_path):
    folder = tmp_path / "extracting-worker"
    _json(folder / "progress.json", _progress())
    with _serve(tmp_path) as port:
        _, overview, _ = _request(port, "/api/overview")
        _, detail, _ = _request(port, "/api/job?id=" + overview["jobs"][0]["id"])
        assert detail["job"]["status"] == "running" and detail["job"]["stage"] == "extract"
        assert detail["job"]["fraction"] == .25 and detail["job"]["completed_layers"] == []
        assert detail["job"]["selection"] == {} and detail["job"]["analysis"] == {}
        for field in ("grid", "ssi", "inference", "openclass", "language"):
            assert detail[field] == []


def test_concurrent_requests_and_nonobject_legacy_state_remain_valid_json(tmp_path):
    _json(tmp_path / "worker" / "progress.json", _progress())
    _json(tmp_path / "malformed" / "progress.json", ["not a progress object"])
    _json(tmp_path / "malformed" / "status.json", {"status": "completed", "model": "legacy"})
    with _serve(tmp_path) as port:
        _, overview, _ = _request(port, "/api/overview")
        key = next(job["id"] for job in overview["jobs"] if job["relative_path"] == "worker")
        paths = ["/api/overview", "/api/job?id=" + key] * 12
        with ThreadPoolExecutor(max_workers=6) as clients:
            responses = list(clients.map(lambda path: _request(port, path), paths))
        assert all(status == 200 for status, _, _ in responses)
        assert all(isinstance(body, dict) for _, body, _ in responses)


def test_pooling_summary_streams_large_file_and_reuses_unchanged_counts(tmp_path, monkeypatch):
    path = tmp_path / "pooling.csv"
    with path.open("w", encoding="utf8") as stream:
        stream.write("item_id,valid,status\n")
        for index in range(81218):
            stream.write(f"item_{index},{'True' if index % 2 else 'False'},ok\n")
    data = DashboardData(tmp_path)
    tracemalloc.start()
    try:
        summary = data._pooling_summary(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert summary == {"items": 81218, "valid": 40609, "invalid": 40609}
    assert peak < 2_000_000  # Materializing 81k row dictionaries exceeds this by an order of magnitude.
    assert not data._csv_cache
    original_open = Path.open
    reads = []
    def watched_open(self, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == path and "r" in mode: reads.append(self)
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", watched_open)
    assert data._pooling_summary(path) == summary
    assert reads == []
    with path.open("a", encoding="utf8") as stream: stream.write("new,True,ok\n")
    assert data._pooling_summary(path)["items"] == 81219
    assert len(reads) == 1


def test_group_filter_precedes_row_limit_and_has_separate_cache_entry(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("group,layer,R\n" + "cultural,0,0.9\n" * 6 +
                    "all,1,0.6\n" + "rural,1,0.8\n" * 6 + "all,2,0.7\nall,3,0.8\n", encoding="utf8")
    data = DashboardData(tmp_path)
    assert [row["group"] for row in data.csv(path, limit=2)] == ["cultural", "cultural"]
    assert [row["layer"] for row in data.csv(path, limit=2, group_all=True)] == [1, 2]
    for index in range(25):
        another = tmp_path / f"grid_{index}.csv"
        another.write_text("group,R\nall,0.5\n", encoding="utf8")
        data.csv(another,group_all=True)
    assert len(data._csv_cache) <= 20
