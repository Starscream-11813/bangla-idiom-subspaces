#!/usr/bin/env bash
# Run the follow-up experiments X1-X19 on the GPU machine, group by group, each behind its reproduction gate.
#
#   bash scripts/run_experiments.sh CAMPAIGN [OUT]
#
# CAMPAIGN is the discovery campaign written by run_research.py (read-only), with its prospective follow-up in
# CAMPAIGN/paper_followup_seed1_20261021 (scripts/run_replication.sh). OUT defaults to CAMPAIGN/followup_experiments.
# Set IDIO_PYTHON to the research interpreter and HF_HOME to the model cache the campaign used.
#
# For each group the driver runs preflight, the gate experiment, `verify --check <gate>`, the other experiments and
# a full verify. It stops if a required preflight item is missing or a gate fails; after a group's gate, later
# experiments of that group run even if an earlier one fails. X14 (decoder) runs only when preflight finds a
# complete decoder tree. Safe to re-run: finished cells are restored from OUT. Logs go to OUT/logs/.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
campaign_dir="${1:?usage: run_experiments.sh CAMPAIGN [OUT]}"
out_dir="${2:-$campaign_dir/followup_experiments}"
py="${IDIO_PYTHON:-python}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
mkdir -p "$out_dir/logs"
cd "$repo_dir"
cli() { "$py" run_experiments.py "$@"; }
run() { cli "$@" --campaign "$campaign_dir" --out "$out_dir"; }
stamp() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$out_dir/logs/driver.log"; }

group() {   # group name, gate, then the other experiments in order
  local name="$1" gate="$2"; shift 2
  stamp "preflight $name"
  run preflight --group "$name" 2>&1 | tee "$out_dir/logs/preflight_$name.log"
  stamp "$gate (gate of $name)"
  run "$gate" >> "$out_dir/logs/$gate.log" 2>&1
  cli verify --check "$gate" --out "$out_dir" 2>&1 | tee "$out_dir/logs/verify_after_$gate.log"
  for x in "$@"; do
    if [ "$x" = X14 ] && ! grep -q "decoder tree for .* complete" "$out_dir/logs/preflight_$name.log"; then
      stamp "X14 skipped: preflight found no complete decoder tree (optional)"
      continue
    fi
    stamp "$x"
    if ! run "$x" >> "$out_dir/logs/$x.log" 2>&1; then
      stamp "$x FAILED - see $out_dir/logs/$x.log"
    fi
  done
  stamp "verify $name"
  cli verify --group "$name" --out "$out_dir" 2>&1 | tee "$out_dir/logs/verify_$name.log"
}

group X1-X8 X5 X1 X3 X7 X8 X2 X6 X4
group X9-X14 X10 X12 X13 X9 X11 X14
group X15-X19 X15 X16 X17 X19 X18
du -sh "$repo_dir/results/experiments" | tee -a "$out_dir/logs/driver.log"
