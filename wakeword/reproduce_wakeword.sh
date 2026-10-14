#!/usr/bin/env bash
# Rebuild the "Hello / Hi Elkit" wake word from scratch (docs/03_wakeword_elkit.md). Run from the repo root.
#
#   bash wakeword/reproduce_wakeword.sh            submit the steps as a chain of Slurm jobs
#   bash wakeword/reproduce_wakeword.sh --local    run them one after another here
#
# Needs: bash env/setup.sh whisper (.venv, .venv-audio, .venv-tts, .venv-whisper), python data_prep/download_data.py
# and python data_prep/build_filtered.py (ai231 + noise), then ~60 GB free under $VCM_DATA.
# The released head (job 65684) = train_oww.py recipe + 10.3 h real sentences as negatives (build_sentences.py).
# Everything is CPU except step "el" (Whisper, one GPU). Total several hours, dominated by TTS generation.
# Settings: VCM_DATA, CPU_PART, GPU_PART, GPU_GRES as in reproduce.sh; WORKERS (CPU processes, default 16).
set -euo pipefail
cd "$(dirname "$0")/.."
export VCM_DATA="${VCM_DATA:-$PWD/data}" WORKERS="${WORKERS:-16}" OMP_NUM_THREADS=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
CPU_PART="${CPU_PART:-uperdfi-virtual}" GPU_PART="${GPU_PART:-uperdfi-gpu}" GPU_GRES="${GPU_GRES:-gpu:nvidia_rtx_a6000:1}"
PY=$PWD/.venv/bin/python; PA=$PWD/.venv-audio/bin/python; PT=$PWD/.venv-tts/bin/python; PW=$PWD/.venv-whisper/bin/python
W=wakeword; E=$VCM_DATA/elkit; V=$VCM_DATA/tts_voices
SPW=$PWD/.venv-whisper/lib/python3.9/site-packages/nvidia   # faster-whisper needs the pip CUDA libraries on the path
mkdir -p "$E" logs

STEPS=(
  # name | cpu/gpu | command
  "download|cpu|$PY $W/download_wakeword_data.py && $PY $W/fetch_mswc.py && EXTRA_WORDS=high $PY $W/fetch_mswc.py"
  "eval_hello|cpu|$PT $W/gen_tts.py $V $E/hello_eval/tts && $PT $W/gen_mms.py $E/hello_eval/mms && $PA $W/make_conditions.py $E/hello_eval/tts,$E/hello_eval/mms $E/hello_eval/clips"
  "eval_hi|cpu|export GREETING=hi && $PT $W/gen_tts.py $V $E/hi_eval/tts && $PT $W/gen_mms.py $E/hi_eval/mms && $PA $W/make_conditions.py $E/hi_eval/tts,$E/hi_eval/mms $E/hi_eval/clips && $PA $W/merge_sets.py $E/eval index.csv $E/hello_eval/clips $E/hi_eval/clips"
  "kws_hello|cpu|$PT $W/gen_kws_tts.py $V $E/hello_kws && $PA $W/world_kws.py $E/hello_kws"
  "kws_hi|cpu|export GREETING=hi && $PT $W/gen_kws_tts.py $V $E/hi_kws && $PA $W/world_kws.py $E/hi_kws && $PA $W/merge_sets.py $E/kws index.csv $E/hello_kws $E/hi_kws"
  "el|gpu|LD_LIBRARY_PATH=$SPW/cublas/lib:$SPW/cudnn/lib:\${LD_LIBRARY_PATH:-} $PW $W/cut_el_mls.py $VCM_DATA/mls_es $VCM_DATA/mls_es/el_clips"
  "real|cpu|$PA $W/build_real.py train $E/kws && $PA $W/build_real.py eval $E/eval"
  "sent|cpu|$PY $W/build_sentences.py"
  "train|cpu|$PY -u -W ignore $W/train_oww_sent.py --index index_real.csv --kws-data $E/kws --eval-clips $E/eval --out runs/elkit --tag elkit --sent-share 0.3 --ctx-p 0.25 --workers $WORKERS"
  "score|cpu|$PY $W/score_eval.py runs/elkit/hello_elkit_oww_head.onnx runs/elkit/scores_engine.npz && $PY $W/wake_metrics.py runs/elkit/scores_engine.npz runs/elkit/wake_metrics.json --extra 0.886:2 0.94:2"
)

if [ "${1:-}" = "--local" ]; then
  for s in "${STEPS[@]}"; do IFS='|' read -r name kind cmd <<< "$s"; echo "=== $name"; bash -c "$cmd" 2>&1 | tee "logs/wake_$name.log"; done
  exit 0
fi
declare -A id
deps() {  # which earlier steps each step waits for
  case "$1" in download) echo "";; eval_hello|kws_hello|el|sent) echo "${id[download]}";; eval_hi) echo "${id[eval_hello]}";;
    kws_hi) echo "${id[kws_hello]}";; real) echo "${id[eval_hi]}:${id[kws_hi]}:${id[el]}";; train) echo "${id[real]}:${id[sent]}";; score) echo "${id[train]}";; esac
}
for s in "${STEPS[@]}"; do
  IFS='|' read -r name kind cmd <<< "$s"
  if [ "$kind" = gpu ]; then res=(-p "$GPU_PART" --gres="$GPU_GRES" -c 4 --mem=32G -t 6:00:00)
  else res=(-p "$CPU_PART" -c "$WORKERS" --mem=64G -t 12:00:00); fi
  d=$(deps "$name"); dep=(); [ -n "$d" ] && dep=(-d "afterok:$d")
  id[$name]=$(sbatch --parsable "${res[@]}" "${dep[@]}" -J "elkit-$name" -o "logs/wake_$name-%j.out" --wrap "cd $PWD && $cmd")
  echo "submitted $name: job ${id[$name]}"
done
