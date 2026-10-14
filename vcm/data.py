"""Datasets and the speaker-disjoint train/validation split (docs/01_dataset.md, docs/02_preprocessing.md).

Validation speakers: 15 % of the speakers of the original ai231 main train split, drawn per (source, is_synthetic)
stratum with seed 23101 (at least one speaker when a stratum has >= 2); speakers that also appear in
supplemental_fil always stay in training. The list is computed from the ORIGINAL ai231 download (so filtering cannot
move a speaker) and must equal artifacts/splits/validation_speakers.json.
Training clips = ai231_v2 main train (non-validation speakers) + supplemental_fil + supplemental_synth (train voices);
clips of validation speakers in the supplemental sets are dropped. oversample.csv repeats listed clips.
Test and holdout are the untouched official splits (different speakers from train and validation).
"""
from __future__ import annotations

import csv, hashlib, io, json, random, wave
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch import Tensor
from torch.utils.data import Dataset

from .common import ARTIFACTS, CLASSES, SPLIT_SEED
from .features import MAX_SAMPLES, LogMel, decode_wav

COLS = ["command", "speaker_id", "source", "is_synthetic", "duration_s", "file", "out_of_scope"]


def read(files: Sequence[Path], audio: bool = False) -> pa.Table:
    return pa.concat_tables([pq.read_table(p, columns=COLS + (["audio"] if audio else [])) for p in files])


def eligible(table: pa.Table) -> list[int]:
    """Row indices of in-scope commands (OUT_OF_SCOPE is not a training target of the 19-way intent model)."""
    rows = table.select(COLS).to_pylist()
    for r in rows:
        assert r["command"] in CLASSES + ["OUT_OF_SCOPE"], r["command"]
        assert bool(r["out_of_scope"]) == (r["command"] == "OUT_OF_SCOPE")
        if r["command"] in CLASSES: assert r["speaker_id"], "Missing in-scope speaker ID"
    return [i for i, r in enumerate(rows) if r["command"] in CLASSES]


def stats(table: pa.Table, indices: Sequence[int]) -> dict:
    rows = table.select(COLS).take(pa.array(indices, type=pa.int64())).to_pylist()
    return {"count": len(rows), "hours": sum(r["duration_s"] for r in rows) / 3600,
            "speakers": len({r["speaker_id"] for r in rows if r["speaker_id"]}),
            "human": sum(r["is_synthetic"] == 0 for r in rows), "synthetic": sum(r["is_synthetic"] == 1 for r in rows),
            "class_counts": dict(sorted(Counter(r["command"] for r in rows).items()))}


def validation_speakers(ai231: Path) -> list[str]:
    """Seeded speaker-level 15 % validation split on the ORIGINAL ai231 train split (see module docstring)."""
    main = read(sorted((ai231 / "data").glob("train-*.parquet")))
    fil = read(sorted((ai231 / "supplemental_fil").glob("train-*.parquet")))
    speakers = main["speaker_id"].to_pylist(); fil_speakers = set(fil["speaker_id"].to_pylist())
    source, synthetic = main["source"].to_pylist(), main["is_synthetic"].to_pylist()
    strata: dict[tuple, set] = {}
    for i in eligible(main):
        if speakers[i] in fil_speakers: continue
        strata.setdefault((source[i], synthetic[i]), set()).add(speakers[i])
    rng = random.Random(SPLIT_SEED); chosen: set[str] = set()
    for _, group in sorted(strata.items()):
        group = sorted(group); rng.shuffle(group)
        if len(group) > 1: chosen.update(group[:max(1, round(len(group) * .15))])
    return sorted(chosen)


def check_validation_speakers(data: Path, frozen_name: str = "validation_speakers.json") -> list[str]:
    computed = validation_speakers(data)
    frozen = json.loads((ARTIFACTS / "splits" / frozen_name).read_text())["validation_speaker_ids"]
    assert computed == frozen, f"validation speakers differ from artifacts/splits/{frozen_name}"
    return computed


SUBSETS = (("data", 2), ("supplemental_fil", 3), ("supplemental_synth", 1))  # (folder, expected train shards)


def prepare_intent(v2: Path, ai231: Path, oversample: Path | None = None):
    """-> (table with audio, train indices (with oversampling repeats), validation indices, details)."""
    files, parts = {}, []
    for sub, n in SUBSETS:
        files[sub] = sorted((v2 / sub).glob("train-*.parquet"))
        assert len(files[sub]) == n, (sub, files[sub])
        parts.append(read(files[sub], True))
    table = pa.concat_tables([p.cast(parts[0].schema) for p in parts])
    offsets = [0, len(parts[0]), len(parts[0]) + len(parts[1])]
    speakers = table["speaker_id"].to_pylist(); names = table["file"].to_pylist()

    vs = set(check_validation_speakers(ai231))
    vi, ti, excluded = [], [], 0
    for k in range(len(SUBSETS)):
        for i in eligible(parts[k]):
            g = offsets[k] + i
            if speakers[g] in vs:
                if k == 0: vi.append(g)
                else: excluded += 1
            else:
                ti.append(g)
    unique_ti = list(ti)
    where = {(SUBSETS[k][0], names[g]): g for k in range(len(SUBSETS)) for g in unique_ti
             if offsets[k] <= g < offsets[k] + len(parts[k])}
    over_file = oversample or v2 / "oversample.csv"
    over = list(csv.DictReader(over_file.open()))
    extra, missing = [], 0
    for r in over:
        g = where.get((r["subset"], r["file"]))
        if g is None: missing += 1; continue  # clip of a validation speaker
        extra += [g] * (int(r["repeat"]) - 1)
    ti = unique_ti + extra

    ts = {speakers[g] for g in unique_ti}
    assert not ts & vs, "train/validation speaker overlap"
    hashes = lambda inds: {hashlib.sha256(table["audio"][i].as_py()["bytes"]).hexdigest() for i in inds}
    assert not hashes(unique_ti) & hashes(vi), "train/validation duplicate audio"
    detail = {"train": stats(table, ti), "train_unique": stats(table, unique_ti), "validation": stats(table, vi),
              "oversample_file": str(over_file), "oversample_rows": len(over), "oversample_extra_indices": len(extra),
              "oversample_rows_in_validation": missing, "supplemental_clips_of_validation_speakers_excluded": excluded,
              "train_indices": ti, "validation_indices": vi, "train_speaker_ids": sorted(ts), "validation_speaker_ids": sorted(vs)}
    return table, ti, vi, detail


class IntentDataset(Dataset):
    def __init__(self, table: pa.Table, indices: Sequence[int], mapping: dict[str, int], augmentor=None) -> None:
        self.audio = table.column("audio"); self.commands = table.column("command")
        self.indices = np.asarray(indices, dtype=np.int64); self.mapping = mapping; self.augmentor = augmentor
        self.front_end = LogMel()

    def __len__(self) -> int: return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, int]:
        row = int(self.indices[item])
        waveform = decode_wav(self.audio[row].as_py()["bytes"])[..., :MAX_SAMPLES]
        if self.augmentor is not None: waveform = self.augmentor(waveform)
        return self.front_end(waveform), self.mapping[self.commands[row].as_py()]


def official_split(ai231: Path, split: str) -> tuple[pa.Table, list[int]]:
    """Untouched test / holdout split and its in-scope row indices."""
    tab = read(sorted((ai231 / "data").glob(f"{split}-*.parquet")), True)
    return tab, [i for i, c in enumerate(tab["command"].to_pylist()) if c in CLASSES]
