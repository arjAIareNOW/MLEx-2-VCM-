#!/usr/bin/env bash
# Create the Python environments used in this repo (run from the repo root).
#   bash env/setup.sh            -> .venv (train)  + .venv-audio + .venv-tts
#   bash env/setup.sh whisper    -> also .venv-whisper (only to re-run the Whisper audit)
set -euo pipefail
cd "$(dirname "$0")/.."
make_env() {  # name python requirements
  [ -x "$1/bin/python" ] || "$2" -m venv "$1"
  "$1/bin/python" -m pip install --upgrade pip
  "$1/bin/python" -m pip install -r "$3"
}
make_env .venv       "${PYTHON39:-python3.9}"  env/requirements-train.txt
make_env .venv-audio "${PYTHON311:-python3.11}" env/requirements-audio.txt
make_env .venv-tts   "${PYTHON311:-python3.11}" env/requirements-tts.txt
if [ "${1:-}" = whisper ]; then make_env .venv-whisper "${PYTHON39:-python3.9}" env/requirements-whisper.txt; fi
echo "Environments ready."
