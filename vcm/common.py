"""Shared helpers: paths, the 19 intent classes, hashing, reproducibility, environment capture."""
from __future__ import annotations

import hashlib, json, os, platform, random, socket
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
# Everything downloaded or built lives under DATA_ROOT (default: <repo>/data). Override with the VCM_DATA env variable.
DATA_ROOT = Path(os.environ.get("VCM_DATA", REPO / "data")).expanduser().resolve()
AI231 = DATA_ROOT / "ai231"                      # Hugging Face snapshot (all configs)
AI231_V2 = DATA_ROOT / "ai231_v2"                # training set built by data_prep/build_ai231_v2.py
SPEECH_NOISE = DATA_ROOT / "noise/speech_commands"  # 6 Speech Commands background-noise WAVs
DEMAND = DATA_ROOT / "noise/demand"              # DEMAND ch01 16 kHz WAVs, one folder per environment
ARTIFACTS = REPO / "artifacts"

CLASSES = ["ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER", "LIGHT_OFF", "LIGHT_ON", "LIST_REMINDERS",
           "MESSAGE", "NEXT", "PAUSE", "PLAY_MUSIC", "STOP", "TEMPERATURE", "TIME", "TIMER", "VOLUME_DOWN", "VOLUME_UP",
           "WEATHER"]
SPLIT_SEED = 23101  # speaker split seed (fixed)


def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_reproducible(seed: int, cpu_threads: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads); torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_environment(device: torch.device) -> dict[str, Any]:
    import pyarrow, torchaudio
    gpu: dict[str, Any] = {"model": None, "count": torch.cuda.device_count(), "memory_bytes": None, "cuda_version": torch.version.cuda}
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        gpu |= {"model": p.name, "memory_bytes": p.total_memory}
    return {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(), "gpu": gpu,
            "python_version": platform.python_version(), "pytorch_version": str(torch.__version__),
            "torchaudio_version": str(torchaudio.__version__), "pyarrow_version": str(pyarrow.__version__),
            "numpy_version": str(np.__version__), "platform": platform.platform()}
