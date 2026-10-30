"""Run independent encoder jobs across visible GPUs; each model owns its output.

Two T4s are two independent 16 GB workers. This never assumes a single 32 GB GPU.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from idiom_subspace.config import Config, model_slug


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--data", required=True)
    p.add_argument("--out", required=True); p.add_argument("--gpus", default="0,1")
    p.add_argument("--cache-dir", action="append", default=[])
    p.add_argument("--cache-only", "--analysis-only", dest="cache_only", action="store_true")
    p.add_argument("--extract-only", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--threads", type=int)
    p.add_argument("--seeds", help="Comma-separated split seeds (replicate refits are opt-in)")
    p.add_argument("--label-seeds", help="Comma-separated taxonomy seeds")
    p.add_argument("--probe-backend", choices=["auto", "torch", "sklearn"])
    p.add_argument("--split-mode", choices=["input_disjoint", "counterpart_closure", "strict", "idiom_only"])
    p.add_argument("--models", help="Comma-separated subset of configured model names")
    p.add_argument("--literal-reviews", help="English literal-gloss CSV (translate-literals output and/or human review)")
    args = p.parse_args()
    cfg = Config.load(args.config)
    if args.threads is not None: cfg.cpu_threads = args.threads
    if args.seeds: cfg.seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    if args.label_seeds: cfg.label_seeds = [int(v) for v in args.label_seeds.split(",") if v.strip()]
    if args.probe_backend: cfg.probe_backend = args.probe_backend
    if args.split_mode: cfg.split_mode = args.split_mode
    if args.models:
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = set(wanted) - {m["name"] for m in cfg.models}
        if unknown: p.error(f"Models not in config: {sorted(unknown)}")
        cfg.models = [m for m in cfg.models if m["name"] in wanted]
    cfg.validate()
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus or len(set(gpus)) != len(gpus): p.error("Provide distinct GPU IDs")
    root = Path(args.out).resolve(); root.mkdir(parents=True, exist_ok=True)
    queue, running, finished = list(cfg.models), {}, []
    entry = Path(__file__).with_name("run_research.py")
    from idiom_subspace.progress import atomic_json, utc_now
    started = {}

    def publish(state="running"):
        # Live queue for the dashboard: planned, running and finished model jobs.
        atomic_json(root / "launcher.json", dict(
            status=state, updated_at=utc_now(), gpus=gpus, config=str(Path(args.config).resolve()),
            seeds=cfg.seeds, label_seeds=cfg.label_seeds,
            planned=[m["name"] for m in cfg.models], queued=[m["name"] for m in queue],
            running=[dict(model=name, gpu=gpu, started_at=started.get(name), pid=getattr(proc, "pid", None))
                     for gpu, (proc, _, name) in running.items()],
            finished=finished))
    try:
        while queue or running:
            for gpu in gpus:
                if gpu in running or not queue: continue
                spec = queue.pop(0)
                out = root / model_slug(spec["name"])
                out.mkdir(parents=True, exist_ok=True)
                logfile = (out / "job.log").open("a", encoding="utf-8")
                env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
                for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                    env[variable] = str(cfg.cpu_threads)
                cmd = [sys.executable, str(entry), "run", "--config", str(Path(args.config).resolve()),
                       "--data", str(Path(args.data).resolve()), "--model", spec["name"], "--out", str(out),
                       "--threads", str(cfg.cpu_threads)]
                for cache in args.cache_dir: cmd.extend(["--cache-dir", str(Path(cache).resolve())])
                for flag in ["cache_only", "extract_only", "no_resume", "no_progress"]:
                    if getattr(args, flag): cmd.append("--" + flag.replace("_", "-"))
                cmd += ["--seeds", ",".join(map(str, cfg.seeds)), "--label-seeds", ",".join(map(str, cfg.label_seeds)),
                        "--split-mode", cfg.split_mode]
                if args.probe_backend: cmd += ["--probe-backend", args.probe_backend]
                if args.literal_reviews: cmd += ["--literal-reviews", str(Path(args.literal_reviews).resolve())]
                try:
                    proc = subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT)
                except BaseException:
                    logfile.close()
                    raise
                running[gpu] = (proc, logfile, spec["name"])
                started[spec["name"]] = utc_now()
                print(f"GPU {gpu}: {spec['name']} -> {out}", flush=True)
                publish()
            for gpu, (proc, logfile, name) in list(running.items()):
                status = proc.poll()
                if status is None: continue
                logfile.close(); finished.append(dict(model=name, gpu=gpu, exit_code=status))
                del running[gpu]
                finished[-1].update(started_at=started.get(name), finished_at=utc_now())
                atomic_json(root / "jobs.json", finished)
                publish()
                print(f"Finished {name}: exit {status}", flush=True)
            if running: time.sleep(1)
    finally:
        # A cancelled launcher must not leave expensive invisible workers running.
        for gpu, (proc, logfile, name) in running.items():
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
            logfile.close()
            finished.append(dict(model=name, gpu=gpu, exit_code=proc.returncode, cancelled=True))
        running.clear()
        atomic_json(root / "jobs.json", finished)
        publish("finished" if not queue else "cancelled")
    # Consolidate planned model jobs and correct the family across all jobs.
    import pandas as pd
    from idiom_subspace.geometry import holm
    from idiom_subspace.runner import planned_primary_tests
    outputs = [root / model_slug(r["model"]) for r in finished if r["exit_code"] == 0]
    inference = [pd.read_csv(folder / "inference_all.csv") for folder in outputs if (folder / "inference_all.csv").exists()]
    if inference:
        merged = pd.concat(inference, ignore_index=True)
        family = planned_primary_tests(cfg)
        for null in ["RAND", "PERM"]: merged[null+"_p_holm_all_jobs"] = holm(merged[null+"_p"].to_numpy(), family_size=family)
        merged.to_csv(root / "inference_all_jobs.csv", index=False)
        atomic_json(root / "inference_design_all_jobs.json", dict(primary_family_size=family,
            observed_primary_pvalues=int(merged.RAND_p.notna().sum()),
            null_draws=cfg.null_draws, minimum_attainable_holm_p=min(1.,family/(cfg.null_draws+1)),
            can_reject_at_alpha=family/(cfg.null_draws+1)<=.05, complete=all(not r["exit_code"] for r in finished)))
    predictions = [pd.read_csv(folder / "predictions_all.csv") for folder in outputs if (folder / "predictions_all.csv").exists()]
    if predictions: pd.concat(predictions, ignore_index=True).to_csv(root / "predictions_all_jobs.csv", index=False)
    raise SystemExit(1 if any(r["exit_code"] for r in finished) else 0)


if __name__ == "__main__": main()
