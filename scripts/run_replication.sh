#!/usr/bin/env bash
# Prospective internal replication on split seed 1 (plan: configs/plans/FOLLOWUP_PLAN.md), on the GPU machine.
#
#   bash scripts/run_replication.sh CAMPAIGN [OUT]
#
# CAMPAIGN is the discovery campaign written by run_research.py; its feature caches are opened read-only.
# OUT defaults to CAMPAIGN/paper_followup_seed1_20261021, the follow-up directory that run_experiments.py reads.
# Set IDIO_PYTHON to the research interpreter and HF_HOME to the model cache the campaign used.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
campaign_dir="${1:?usage: run_replication.sh CAMPAIGN [OUT]}"
followup_dir="${2:-$campaign_dir/paper_followup_seed1_20261021}"
research_python="${IDIO_PYTHON:-python}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p "$followup_dir"
cd "$repo_dir"
exec "$research_python" run_replication.py --campaign "$campaign_dir" --out "$followup_dir" >> "$followup_dir/run.log" 2>&1
