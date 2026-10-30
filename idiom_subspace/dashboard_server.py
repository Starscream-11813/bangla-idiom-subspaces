"""Read-only, loopback-only experiment dashboard; no cloud or Node service.

The server only reads the experiment output tree. It never launches, stops or
edits a run and never serves arbitrary files: the API returns parsed progress
records, summary tables and (optionally) a sanitized tail of ``job.log``.
``--snapshot FILE`` writes the same views into one self-contained HTML file
that can be copied off a remote machine and opened without a server.
"""
import argparse
from collections import OrderedDict
import csv
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import re
import socket
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlparse

ASSETS = Path(__file__).with_name("dashboard_assets")
SKIP_DIRS = {"cache", "checkpoints", "node_modules", ".git", "__pycache__", ".pytest_cache"}
LOG_NAMES = ("job.log", "run.log")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
STALE_SECONDS = 30


def read_json(path, fallback=None):
    try:
        if path.stat().st_size > 4_000_000: return fallback
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError): return fallback


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list): return [json_safe(v) for v in value]
    return value


def _age_seconds(stamp):
    if not stamp: return None
    try:
        observed = datetime.fromisoformat(stamp)
        if observed.tzinfo is None: observed = observed.replace(tzinfo=timezone.utc)
        return max(0, (datetime.now(timezone.utc) - observed).total_seconds())
    except (ValueError, TypeError):
        return None


class DashboardData:
    def __init__(self, root, *, logs=True):
        self.root = Path(root).resolve()
        self.logs = logs
        self._lock = threading.Lock()
        self._jobs, self._orchestrators, self._launchers, self._last_scan = {}, [], [], 0.
        self._csv_cache = OrderedDict()
        self._pooling_cache = {}
        self._log_cache = {}
        self._cache_lock = threading.RLock()

    def scan(self):
        with self._lock:
            if time.monotonic() - self._last_scan < 1: return self._jobs
            jobs, orchestrators, launchers = {}, [], []
            if self.root.exists():
                for directory, dirs, files in os.walk(self.root):
                    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
                    folder = Path(directory)
                    relative = str(folder.relative_to(self.root)).replace("\\", "/") or "."
                    if "launcher.json" in files:
                        launcher = read_json(folder / "launcher.json")
                        if isinstance(launcher, dict):
                            launchers.append(dict(launcher, relative_path=relative,
                                                  heartbeat_age_seconds=_age_seconds(launcher.get("updated_at"))))
                    if "progress.json" not in files and "status.json" not in files: continue
                    state = read_json(folder / "progress.json")
                    status = read_json(folder / "status.json", {})
                    if not isinstance(state, dict): state = None
                    if not isinstance(status, dict): status = {}
                    telemetry_available = bool(state)
                    if not state:
                        state = dict(status=status.get("status", "unknown"), stage="legacy_run",
                                     context={"model": status.get("model", folder.name)},
                                     message="No live telemetry: this run predates progress tracking.")
                    if state.get("context", {}).get("kind") == "orchestrator":
                        age = _age_seconds(state.get("updated_at"))
                        orchestrators.append(dict(
                            relative_path=relative, status=state.get("status"), stage=state.get("stage"),
                            completed=state.get("completed"), total=state.get("total"),
                            plan_jobs=state.get("context", {}).get("plan_jobs"),
                            mode=state.get("context", {}).get("mode"), elapsed_seconds=state.get("elapsed_seconds"),
                            heartbeat_age_seconds=age,
                            stale=state.get("status") == "running" and (age or 0) > STALE_SECONDS,
                            history=state.get("history", [])[-50:], updated_at=state.get("updated_at")))
                        continue
                    key = hashlib.sha256(relative.encode()).hexdigest()[:16]
                    state = dict(state, id=key, relative_path=relative)
                    state["telemetry_available"] = telemetry_available
                    age = _age_seconds(state.get("updated_at"))
                    if age is not None: state["heartbeat_age_seconds"] = age
                    state["heartbeat_unknown"] = state.get("status") == "running" and "heartbeat_age_seconds" not in state
                    state["stale"] = state.get("status") == "running" and state.get("heartbeat_age_seconds", 0) > STALE_SECONDS
                    state["live"] = (state.get("status") == "running" and telemetry_available
                                     and not state["heartbeat_unknown"] and not state["stale"])
                    state["selection"] = read_json(folder / "layer_selection.json", {})
                    state["analysis"] = read_json(folder / "analysis_manifest.json", {})
                    state["pooling"] = self._pooling_summary(folder / "pooling.csv")
                    state["completed_layers"] = sorted(int(p.stem.removeprefix("grid_layer_")) for p in folder.glob("grid_layer_*.csv") if p.stem.removeprefix("grid_layer_").isdigit())
                    state["folder"] = folder
                    jobs[key] = state
            self._jobs, self._orchestrators, self._launchers = jobs, orchestrators, launchers
            self._last_scan = time.monotonic()
            return jobs

    def csv(self, path, limit=50000, *, group_all=False):
        try: stat = path.stat()
        except OSError: return []
        key = (str(path), stat.st_mtime_ns, stat.st_size, limit, group_all)
        with self._cache_lock:
            cached = self._csv_cache.get(key)
            if cached is not None:
                self._csv_cache.move_to_end(key)
                return cached
        rows=[]
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    # Count the rows actually plotted, rather than spending the
                    # limit on subgroup records and dropping later layers.
                    if group_all and row.get("group") != "all": continue
                    cooked={}
                    for k,v in row.items():
                        if k is None: continue
                        if v is None or v.casefold() in {"", "nan", "none", "null", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}: cooked[k]=None
                        elif v in {"True", "False"}: cooked[k]=v=="True"
                        else:
                            try: cooked[k]=float(v) if any(c in v for c in ".eE") else int(v)
                            except (ValueError, TypeError): cooked[k]=v
                    rows.append(cooked)
                    if len(rows)>=limit: break
        except (OSError, csv.Error, UnicodeError): return []
        with self._cache_lock:
            self._csv_cache[key]=rows
            self._csv_cache.move_to_end(key)
            while len(self._csv_cache)>20: self._csv_cache.popitem(last=False)
        return rows

    def _pooling_summary(self,path):
        try: stat = path.stat()
        except OSError: return {}
        key, stamp = str(path), (stat.st_mtime_ns, stat.st_size)
        with self._cache_lock:
            cached = self._pooling_cache.get(key)
            if cached is not None and cached[0] == stamp: return cached[1]
        items, valid = 0, 0
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    items += 1
                    valid += str(row.get("valid", "")).strip().casefold() == "true"
        except (OSError, csv.Error, UnicodeError): return {}
        summary = dict(items=items,valid=valid,invalid=items-valid) if items else {}
        with self._cache_lock:
            # One tiny summary per path; changed files replace their old entry.
            self._pooling_cache[key] = (stamp, summary)
        return summary

    def _log_tail(self, folder, lines=80, max_bytes=65536):
        """Last lines of the nearest job.log inside the root, ANSI/CR sanitized."""
        if not self.logs: return []
        for directory in [folder, *folder.parents]:
            try: directory.relative_to(self.root)
            except ValueError: break
            for name in LOG_NAMES:
                path = directory / name
                try: stat = path.stat()
                except OSError: continue
                key, stamp = str(path), (stat.st_mtime_ns, stat.st_size)
                with self._cache_lock:
                    cached = self._log_cache.get(key)
                    if cached is not None and cached[0] == stamp: return cached[1]
                try:
                    with path.open("rb") as stream:
                        stream.seek(max(0, stat.st_size - max_bytes))
                        data = stream.read(max_bytes)
                except OSError:
                    return []
                out = []
                # Split on newlines only: a carriage return means the terminal overwrote
                # the line (progress bars), so only its last segment was visible.
                for line in data.decode("utf-8", "replace").split("\n"):
                    line = _ANSI.sub("", line.rstrip("\r").split("\r")[-1]).rstrip()
                    if line: out.append(line[:400])
                out = out[-lines:]
                with self._cache_lock:
                    self._log_cache[key] = (stamp, out)
                    while len(self._log_cache) > 50: self._log_cache.pop(next(iter(self._log_cache)))
                return out
        return []

    def summary(self, states):
        """Whole-run counts and a labelled, rough remaining-time estimate."""
        planned = sum(o.get("plan_jobs") or 0 for o in self._orchestrators)
        launcher_planned = sum(len(l.get("planned") or []) for l in self._launchers)
        remaining = [s.get("milestones", {}).get("layer", {}).get("eta_seconds") for s in states if s.get("live")]
        remaining = [r for r in remaining if isinstance(r, (int, float))]
        return dict(planned_jobs=max(planned, launcher_planned) or None,
                    queued=sum(len(l.get("queued") or []) for l in self._launchers),
                    longest_running_layer_eta_seconds=max(remaining) if remaining else None,
                    eta_kind="layer_milestone_estimate" if remaining else None)

    def overview(self):
        jobs=self.scan()
        states=[{k:v for k,v in value.items() if k!="folder"} for value in jobs.values()]
        return json_safe(dict(root=str(self.root), updated_at=datetime.now(timezone.utc).isoformat(),
                              jobs=states, total=len(states), running=sum(bool(s.get("live")) for s in states),
                              completed=sum(s.get("status")=="completed" for s in states),
                              failed=sum(s.get("status") in {"failed","interrupted"} for s in states),
                              orchestrators=self._orchestrators, launchers=self._launchers,
                              summary=self.summary(states)))

    def detail(self,key):
        value=self.scan().get(key)
        if not value:return None
        folder=value["folder"]
        grids=[]
        # Per-layer files are atomic; read only all-group rows for the visuals.
        for p in sorted(folder.glob("grid_layer_*.csv")):
            grids.extend(self.csv(p,group_all=True))
        split=read_json(folder.parent/"split_audit.json",{})
        if not split:split=read_json(folder.parent.parent/"split_audit.json",{})
        label=read_json(folder.parent/"label_audit.json",{})
        if not label:label=read_json(folder.parent.parent/"label_audit.json",{})
        split_view={k:split.get(k) for k in ["mode","counts","components","largest_component","largest_fraction",
                                              "coverage_warning","cross_split_identical_inputs","form_variant_links",
                                              "overlapping_counterparts"] if k in split}
        result=dict(job={k:v for k,v in value.items() if k!="folder"},grid=grids,
                    ssi=self.csv(folder/"ssi_all_layers.csv",group_all=True), inference=self.csv(folder/"S3_inference.csv",group_all=True),
                    openclass=self.csv(folder/"S4_openclass.csv"),language=self.csv(folder/"S2_language.csv"),
                    baselines=self.csv(folder/"S3_baselines.csv",limit=2000),
                    contrasts=self.csv(folder/"S3_contrasts.csv",limit=2000),
                    asymmetry=self.csv(folder/"S3_asymmetry.csv",limit=2000),
                    fls=self.csv(folder/"S4_FLS.csv",limit=5000),
                    split=split_view, labels={k:label.get(k) for k in ["backend","effective_k","literal_english_available"]},
                    coverage=read_json(folder/"coverage.json",[]),
                    log_tail=self._log_tail(folder))
        return json_safe(result)


def export_snapshot(root, destination, *, logs=True):
    """Write one self-contained HTML file with the current overview and job details."""
    data = DashboardData(root, logs=logs)
    overview = data.overview()
    details = {job["id"]: data.detail(job["id"]) for job in overview["jobs"]}
    payload = json.dumps(json_safe(dict(overview=overview, details=details,
                                        exported_at=datetime.now(timezone.utc).isoformat())),
                         ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = (ASSETS / "index.html").read_text(encoding="utf-8")
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    js = (ASSETS / "app.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    html = html.replace('<link rel="stylesheet" href="/style.css">', f"<style>\n{css}\n</style>")
    # Inline scripts cannot be deferred: run them after the page body has been parsed.
    html = html.replace('<script src="/app.js" defer></script>', "")
    html = html.replace("</body>", f"<script>window.__OBSERVATORY_SNAPSHOT__ = {payload};</script>\n<script>\n{js}\n</script>\n</body>")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    return destination


class LocalDashboardServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can bind over a different app's live listener.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket,"SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


def make_server(root,port=8787,*,logs=True):
    data=DashboardData(root,logs=logs)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            url=urlparse(self.path)
            if url.path=="/api/overview":return self.respond_json(data.overview())
            if url.path=="/api/job":
                key=parse_qs(url.query).get("id",[""])[0]
                result=data.detail(key)
                return self.respond_json(result or {"error":"Unknown experiment"},200 if result else 404)
            names={"/":"index.html","/index.html":"index.html","/app.js":"app.js","/style.css":"style.css"}
            if url.path not in names:return self.respond_json({"error":"Not found"},404)
            path=ASSETS/names[url.path]
            try:body=path.read_bytes()
            except OSError:return self.respond_json({"error":"Dashboard assets missing"},500)
            kind="text/html; charset=utf-8" if path.suffix==".html" else "text/javascript; charset=utf-8" if path.suffix==".js" else "text/css; charset=utf-8"
            self.respond(body,kind)
        def respond_json(self,value,status=200):
            self.respond(json.dumps(json_safe(value),allow_nan=False).encode(),"application/json; charset=utf-8",status)
        def respond(self,body,kind,status=200):
            self.send_response(status);self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)
    server=LocalDashboardServer(("127.0.0.1",port),Handler)
    server.daemon_threads=True
    return server


def main(argv=None):
    parser=argparse.ArgumentParser(description="Local, read-only idiom-subspace experiment dashboard")
    parser.add_argument("--root",required=True,help="New output directory or parent of GPU job directories")
    parser.add_argument("--port",type=int,default=8787)
    parser.add_argument("--no-logs",action="store_true",help="Do not show the tail of job.log files")
    parser.add_argument("--snapshot",metavar="FILE",help="Write a self-contained HTML snapshot and exit")
    args=parser.parse_args(argv)
    if args.snapshot:
        path=export_snapshot(args.root,args.snapshot,logs=not args.no_logs)
        print(f"Snapshot written to {path.resolve()} (open it in any browser; it does not refresh)",flush=True)
        return
    try:server=make_server(args.root,args.port,logs=not args.no_logs)
    except OSError as error:
        parser.exit(2,f"Cannot listen on 127.0.0.1:{args.port}: {error}. Choose another --port.\n")
    print(f"Dashboard: http://127.0.0.1:{server.server_port}  (reading {Path(args.root).resolve()})",flush=True)
    try:server.serve_forever(poll_interval=.3)
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=="__main__":main()
