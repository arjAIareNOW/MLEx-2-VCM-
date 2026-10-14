#!/usr/bin/env bash
# One-command reproduction of the intent and slot models (data download -> preprocessing -> training -> ONNX -> checks).
#
#   bash reproduce.sh            submit every step as a chain of Slurm jobs (each waits for the previous one)
#   bash reproduce.sh --local    run the same steps one after another on this machine (needs one CUDA GPU)
#   add --from STEP to start at a later step (download, build_data, train_intent, train_slot, summary)
#
# Settings (environment variables, all optional):
#   VCM_DATA      where data is downloaded / built            (default: ./data, ~25 GB)
#   RUNS          where trained models are written            (default: ./runs)
#   CPU_PART      Slurm partition for CPU steps               (default: uperdfi-virtual)
#   GPU_PART      Slurm partition for GPU steps               (default: uperdfi-gpu)
#   GPU_GRES      Slurm GPU request                           (default: gpu:nvidia_a100_80gb_pcie:1)
#   INTENT_SEED   training seed of the intent model           (default: 23102, the seed of the released model)
# Environments: run `bash env/setup.sh` first (creates .venv, .venv-audio, .venv-tts).
# The wake word has its own pipeline: wakeword/reproduce_wakeword.sh (CPU only, several hours, ~60 GB).
set -euo pipefail
cd "$(dirname "$0")"
export VCM_DATA="${VCM_DATA:-$PWD/data}" RUNS="${RUNS:-$PWD/runs}"
CPU_PART="${CPU_PART:-uperdfi-virtual}" GPU_PART="${GPU_PART:-uperdfi-gpu}" GPU_GRES="${GPU_GRES:-gpu:nvidia_a100_80gb_pcie:1}"
SEED="${INTENT_SEED:-23102}"
PY="$PWD/.venv/bin/python"
mkdir -p "$VCM_DATA" "$RUNS" logs

STEPS=(
  "download|cpu|$PY data_prep/download_data.py"
  "build_data|cpu|$PY data_prep/build_filtered.py && $PY data_prep/build_ai231_v2.py && $PY data_prep/numerals_subset.py"
  "train_intent|gpu|$PY -u -W ignore -m vcm.train_intent --output $RUNS/intent_conformer --seed $SEED --workers 4 --cpu-threads 1 && $PY -W ignore -m vcm.export_onnx intent $RUNS/intent_conformer && $PY -W ignore -m vcm.reject_threshold $RUNS/intent_conformer"
  "train_slot|gpu|$PY -u -W ignore -m vcm.train_slot --output $RUNS/slot_conformer --workers 4 && $PY -W ignore -m vcm.export_onnx slot $RUNS/slot_conformer"
  "summary|cpu|$PY tools/summarize_runs.py $RUNS"
)

LOCAL=0; FROM=download
while [ $# -gt 0 ]; do case "$1" in --local) LOCAL=1;; --from) FROM="$2"; shift;; *) echo "unknown option $1"; exit 1;; esac; shift; done
started=0; build_id=""; intent_id=""
skip() { [ "$1" = "$FROM" ] && started=1; [ $started = 0 ]; }

if [ $LOCAL = 1 ]; then
  for s in "${STEPS[@]}"; do
    IFS='|' read -r name kind cmd <<< "$s"; skip "$name" && continue
    echo "=== $name"; bash -c "$cmd" 2>&1 | tee "logs/$name.log"
  done
  exit 0
fi

prev=""
for s in "${STEPS[@]}"; do
  IFS='|' read -r name kind cmd <<< "$s"; skip "$name" && continue
  if [ "$kind" = gpu ]; then res=(-p "$GPU_PART" --gres="$GPU_GRES" -c 5 --mem=48G -t 12:00:00)
  else res=(-p "$CPU_PART" -c 8 --mem=48G -t 6:00:00); fi
  dep=(); [ -n "$prev" ] && dep=(-d "afterok:$prev")
  # train_intent and train_slot are independent: both wait for build_data
  [ "$name" = train_slot ] && { dep=(); [ -n "$build_id" ] && dep=(-d "afterok:$build_id"); }
  [ "$name" = summary ] && dep=(-d "afterok:$prev${intent_id:+:$intent_id}")
  id=$(sbatch --parsable "${res[@]}" "${dep[@]}" -J "vcm-$name" -o "logs/$name-%j.out" \
       --wrap "cd $PWD && export OMP_NUM_THREADS=1 && $cmd")  # (no --export=...: it breaks GPU jobs on some Slurm setups)
  echo "submitted $name: job $id"
  [ "$name" = build_data ] && build_id=$id
  [ "$name" = train_intent ] && intent_id=$id
  prev=$id
done
echo "Follow with: squeue -u \$USER ; results: $RUNS/*/config.json and $RUNS/summary.md"
