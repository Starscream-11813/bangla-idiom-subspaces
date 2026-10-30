"""Terminal status line and atomic, process-local progress records for the dashboard.

Each model job owns one ``progress.json``; there is no shared mutable status
across workers. ``fraction`` describes the named stage only. Two estimates are
added and labelled as such: a stage ETA from recent throughput, and an ETA for
repeated milestones (e.g. layers) from their mean completed duration.

Terminal echo: on an interactive terminal a single status line is rewritten in
place; when output is redirected (log files, ``nohup``, notebooks) a compact
timestamped line is printed at most every ``echo_interval`` seconds and at each
stage change, so logs stay readable.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import uuid


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str)
    temp.write_text(text, encoding="utf-8")
    try:
        atomic_replace(temp, path)
    finally:
        try: temp.unlink(missing_ok=True)
        except OSError: pass


def atomic_replace(source, destination):
    """Atomically publish an artifact, including while a Windows reader polls it."""
    for attempt in range(12):
        try:
            os.replace(source, destination)
            return
        except PermissionError as error:
            # Keep the previous complete artifact visible during sharing locks.
            if getattr(error, "winerror", None) not in {5, 32, 33} or attempt == 11:
                raise
            time.sleep(min(.005 * (2 ** attempt), .05))


def format_duration(seconds):
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def _interactive(stream):
    try:
        return stream.isatty() and os.environ.get("TERM") != "dumb"
    except (AttributeError, ValueError):
        return False


class _LogIter:
    """Periodic one-line progress for non-interactive output (log files)."""

    def __init__(self, iterable, desc, total, interval, stream):
        self.iterable, self.desc, self.total = iterable, desc, total
        self.interval, self.stream = interval, stream

    def __iter__(self):
        start = last = time.monotonic()
        count = 0
        for value in self.iterable:
            yield value
            count += 1
            now = time.monotonic()
            if now - last >= self.interval or (self.total and count == self.total):
                rate = count / max(now - start, 1e-9)
                eta = (self.total - count) / rate if self.total and rate > 0 else None
                total = f"/{self.total}" if self.total else ""
                print(f"[{datetime.now():%H:%M:%S}] {self.desc}: {count}{total} "
                      f"| {rate:.2f}/s | ETA {format_duration(eta)}", file=self.stream, flush=True)
                last = now


def progress_iter(iterable, desc="", total=None, enabled=True, interval=30, **kwargs):
    """tqdm on an interactive terminal, periodic log lines otherwise.

    tqdm is optional at import time; plain iteration remains supported.
    """
    if not enabled:
        return iterable
    if total is None:
        try: total = len(iterable)
        except TypeError: total = None
    if not _interactive(sys.stderr):
        return _LogIter(iterable, desc, total, interval, sys.stderr)
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, disable=False, **kwargs)
    except ImportError:
        return _LogIter(iterable, desc, total, interval, sys.stderr)


class Progress:
    """Heartbeat-backed progress journal with an optional terminal status line."""

    def __init__(self, path, *, enabled=True, heartbeat_seconds=3, echo=False, echo_interval=30,
                 stream=None, rate_window_seconds=2., **context):
        self.path, self.enabled = Path(path), enabled
        self._lock, self._stop = threading.RLock(), threading.Event()
        self._start = self._stage_start = time.monotonic()
        self._rate, self._rate_mark = None, None
        self._rate_window = float(rate_window_seconds)
        self._milestones = {}
        self._echo = bool(echo) and enabled
        self._echo_interval = float(echo_interval)
        self._stream = stream or sys.stderr
        self._tty = _interactive(self._stream)
        self._last_echo, self._echo_stage = 0., None
        self._base_context = dict(context)
        self._state = dict(schema_version=2, run_id=uuid.uuid4().hex,
                           status="running", stage="starting", started_at=utc_now(),
                           stage_started_at=utc_now(), pid=os.getpid(), completed=0,
                           total=None, unit="tasks", context=context,
                           message="Preparing experiment", history=[], milestones={})
        self._thread = None
        if not self.enabled:
            return
        self._flush()
        self._thread = threading.Thread(target=self._heartbeat, args=(heartbeat_seconds,), daemon=True)
        self._thread.start()

    # -- estimates -------------------------------------------------------------------------
    def _observe_rate(self, completed):
        """Exponentially weighted throughput of the current stage (units per second).

        Rates are measured over windows of at least two seconds so that units
        completing in a burst (e.g. restored checkpoints) do not produce
        absurd instantaneous throughput.
        """
        now = time.monotonic()
        if self._rate_mark is None:
            self._rate_mark = (now, completed)
            return
        then, before = self._rate_mark
        elapsed, delta = now - then, completed - before
        if delta > 0 and elapsed >= self._rate_window and elapsed > 0:
            instant = delta / elapsed
            weight = min(1., elapsed / 60.)        # ~1-minute memory
            self._rate = instant if self._rate is None else (1 - weight) * self._rate + weight * instant
            self._rate_mark = (now, completed)

    def _estimates(self, state):
        n, total = state.get("completed"), state.get("total")
        state["fraction"] = min(1, max(0, n / total)) if total and n is not None else None
        rate = self._rate
        if rate is None and n and n > 0 and (state.get("stage_elapsed_seconds") or 0) >= max(self._rate_window, 1e-9):
            rate = n / state["stage_elapsed_seconds"]
        rate = rate if rate and rate > 0 else None
        state["rate_per_second"] = rate
        state["eta_seconds"] = ((total - n) / rate) if (total and n is not None and rate and n < total) else None
        state["eta_kind"] = "stage_throughput_estimate" if state["eta_seconds"] is not None else None
        milestones = {}
        for name, info in self._milestones.items():
            done, total_m, durations = info["done"], info["total"], info["durations"]
            mean = sum(durations) / len(durations) if durations else None
            remaining = None
            if mean is not None and total_m:
                running = time.monotonic() - info["last"]
                remaining = max(0., mean * (total_m - done) - min(running, mean))
            milestones[name] = dict(done=done, total=total_m, mean_seconds=mean, eta_seconds=remaining,
                                    durations=durations[-50:], label=info.get("label"))
        state["milestones"] = milestones

    # -- persistence -----------------------------------------------------------------------
    def _flush(self):
        if not self.enabled:
            return
        now = time.monotonic()
        state = dict(self._state)
        state.update(updated_at=utc_now(), elapsed_seconds=now - self._start,
                     stage_elapsed_seconds=now - self._stage_start)
        self._estimates(state)
        atomic_json(self.path, state)
        self._render(state)

    def _render(self, state, force=False):
        if not self._echo:
            return
        now = time.monotonic()
        stage_changed = state["stage"] != self._echo_stage
        if not (force or stage_changed or self._tty and now - self._last_echo >= .5
                or now - self._last_echo >= self._echo_interval):
            return
        self._last_echo, self._echo_stage = now, state["stage"]
        ctx = state.get("context", {})
        who = " ".join(str(v) for v in [ctx.get("model"),
                                        f"split{ctx['seed']}" if ctx.get("seed") is not None else None,
                                        f"tax{ctx['label_seed']}" if ctx.get("label_seed") is not None else None] if v)
        parts = [f"[{datetime.now():%H:%M:%S}]", who or "job"]
        layers = state["milestones"].get("layer")
        if layers and layers.get("total"):
            parts.append(f"layer {layers['done']}/{layers['total']}")
        if ctx.get("layer") is not None:
            parts.append(f"(L{ctx['layer']})")
        count = state.get("completed")
        if state.get("total"):
            parts.append(f"{state['stage']} {count:,}/{state['total']:,} {state.get('unit', '')} "
                         f"({100 * (state['fraction'] or 0):.0f}%)")
        else:
            parts.append(str(state["stage"]))
        if state.get("rate_per_second"):
            parts.append(f"{state['rate_per_second']:.2f}/s")
        if state.get("eta_seconds") is not None:
            parts.append(f"stage ETA {format_duration(state['eta_seconds'])}")
        if layers and layers.get("eta_seconds") is not None:
            parts.append(f"layers ETA {format_duration(layers['eta_seconds'])}")
        parts.append(f"elapsed {format_duration(state.get('elapsed_seconds'))}")
        # Redirected output (log files) stays ASCII so any reader/encoding shows it cleanly.
        separator = " · " if self._tty else " | "
        try:
            separator.encode(getattr(self._stream, "encoding", None) or "utf-8")
        except (UnicodeEncodeError, LookupError):
            separator = " | "
        line = separator.join(parts)
        try:
            if self._tty:
                width = shutil.get_terminal_size((120, 20)).columns
                self._stream.write("\r" + line[:width - 1].ljust(width - 1))
                if state.get("status") != "running":
                    self._stream.write("\n")
            else:
                self._stream.write(line + "\n")
            self._stream.flush()
        except (OSError, ValueError):
            self._echo = False

    def _heartbeat(self, delay):
        while not self._stop.wait(max(.05, delay)):
            with self._lock:
                if self._state["status"] == "running": self._flush()

    # -- public API ------------------------------------------------------------------------
    def update(self, stage=None, completed=None, total=None, message=None, unit=None, **context):
        if not self.enabled:
            return
        with self._lock:
            previous = self._state["stage"]
            if stage is not None and stage != previous:
                self._state["history"] = (self._state["history"] + [dict(
                    stage=previous, duration_seconds=time.monotonic()-self._stage_start,
                    completed=self._state.get("completed"), total=self._state.get("total"))])[-50:]
                self._stage_start = time.monotonic()
                self._rate, self._rate_mark = None, None
                self._state.update(stage=stage, stage_started_at=utc_now(), completed=0, total=None,
                                   unit={"extract":"items", "cache_validation":"items", "nulls":"draws",
                                         "layer_complete":"layers"}.get(stage,"tasks"))
                # Task details belong to a stage; keep the job identity and the current layer.
                layer = self._state["context"].get("layer")
                self._state["context"] = dict(self._base_context)
                if layer is not None: self._state["context"]["layer"] = layer
            if completed is not None:
                self._state["completed"] = int(completed)
                self._observe_rate(int(completed))
            if total is not None: self._state["total"] = int(total)
            if message is not None: self._state["message"] = str(message)
            if unit is not None: self._state["unit"] = str(unit)
            self._state["context"].update(context)
            self._flush()

    def milestone(self, name, done, total, label=None):
        """Record completion of a repeated unit (e.g. layer ``done`` of ``total``)."""
        if not self.enabled:
            return
        with self._lock:
            now = time.monotonic()
            info = self._milestones.setdefault(name, dict(done=0, total=total, durations=[], last=self._start,
                                                          label=label))
            if done > info["done"]:
                info["durations"].append(now - info["last"])
                info["last"] = now
            info.update(done=int(done), total=int(total) if total else None, label=label or info.get("label"))
            self._flush()

    def start_milestone(self, name, total, label=None):
        """Start timing a repeated unit without counting a completion."""
        if not self.enabled:
            return
        with self._lock:
            info = self._milestones.setdefault(name, dict(done=0, total=total, durations=[], label=label,
                                                          last=time.monotonic()))
            info["last"] = time.monotonic()
            info["total"] = int(total) if total else None

    def close(self, status="completed", message=None):
        if not self.enabled:
            return
        self._stop.set()
        with self._lock:
            self._state["status"] = status
            if message is not None: self._state["message"] = str(message)
            self._flush()
        if self._thread is not None and threading.current_thread() is not self._thread: self._thread.join(timeout=2)

    def __enter__(self): return self

    def __exit__(self, kind, value, traceback):
        self.close("interrupted" if kind is KeyboardInterrupt else "failed" if kind else "completed",
                   str(value) if kind else None)
