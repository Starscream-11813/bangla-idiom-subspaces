"""Launcher lifecycle and output-family checks without GPU workers."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import launch_gpu_jobs as launcher
from idiom_subspace.config import Config


def _args(tmp_path, monkeypatch, cfg, *extra):
    config=tmp_path/"config.json"
    config.write_text(json.dumps(cfg.as_dict()))
    data=tmp_path/"data.json";data.write_text("[]")
    output=tmp_path/"jobs"
    monkeypatch.setattr(sys,"argv",["launch_gpu_jobs.py","--config",str(config),"--data",str(data),
        "--out",str(output),"--gpus","0,1",*extra])
    return output


def test_launcher_forwards_cache_flags_and_corrects_only_current_jobs(tmp_path,monkeypatch):
    cfg=Config(models=[{"name":"first/model"},{"name":"second/model"}],seeds=[0],label_seeds=[0])
    output=_args(tmp_path,monkeypatch,cfg,"--cache-dir",str(tmp_path/"old"),"--cache-only","--threads","3")
    stale=output/"unrelated";stale.mkdir(parents=True)
    pd.DataFrame([dict(RAND_p=1e-8,PERM_p=1e-8)]).to_csv(stale/"inference_all.csv",index=False)
    calls=[]
    class Done:
        returncode=0
        def poll(self):return 0
    def spawn(cmd,**kwargs):
        calls.append((cmd,kwargs))
        dest=Path(cmd[cmd.index("--out")+1])
        pd.DataFrame([dict(RAND_p=.001,PERM_p=np.nan)]).to_csv(dest/"inference_all.csv",index=False)
        return Done()
    monkeypatch.setattr(launcher.subprocess,"Popen",spawn)
    with pytest.raises(SystemExit) as result:launcher.main()
    assert result.value.code==0 and len(calls)==2
    for cmd,kwargs in calls:
        assert "--cache-only" in cmd and cmd[cmd.index("--cache-dir")+1]==str((tmp_path/"old").resolve())
        assert kwargs["env"]["OMP_NUM_THREADS"]=="3" and kwargs["env"]["OPENBLAS_NUM_THREADS"]=="3"
        assert kwargs["stdout"].closed
    merged=pd.read_csv(output/"inference_all_jobs.csv")
    assert len(merged)==2  # unrelated historical outputs never enter this family
    np.testing.assert_allclose(merged.RAND_p_holm_all_jobs,.004)  # four planned tests, two missing
    design=json.loads((output/"inference_design_all_jobs.json").read_text())
    assert design["primary_family_size"]==4 and design["observed_primary_pvalues"]==2


def test_cancelled_launcher_terminates_and_reaps_its_worker(tmp_path,monkeypatch):
    cfg=Config(models=[{"name":"first/model"}],seeds=[0],label_seeds=[0])
    output=_args(tmp_path,monkeypatch,cfg)
    class Running:
        returncode=None
        terminated=False
        waited=False
        def poll(self):return self.returncode
        def terminate(self):self.terminated=True;self.returncode=-15
        def wait(self,timeout=None):self.waited=True;return self.returncode
    worker=Running()
    monkeypatch.setattr(launcher.subprocess,"Popen",lambda *a,**k:worker)
    def interrupt(_delay):raise KeyboardInterrupt()
    monkeypatch.setattr(launcher.time,"sleep",interrupt)
    with pytest.raises(KeyboardInterrupt):launcher.main()
    assert worker.terminated and worker.waited
    record=json.loads((output/"jobs.json").read_text())[0]
    assert record["cancelled"] and record["exit_code"]==-15
