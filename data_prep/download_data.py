"""Download every dataset the intent and slot models need, pinned to exact revisions and checked with SHA-256.

  ai231            Hugging Face airimonda/ai231-me2-voice-commands @ 5abbe539 (all configs: data, supplemental_fil,
                   supplemental_synth, synthetic_negatives)          -> $VCM_DATA/ai231            (~5 GB)
  DEMAND           Hugging Face philgzl/demand @ 29f46e61, 16 kHz shards, decoded to ch01 WAVs
                                                                      -> $VCM_DATA/noise/demand     (~2.7 GB)
  Speech Commands  only the 6 files of _background_noise_ from the v0.02 archive (download.tensorflow.org, CC-BY-4.0;
                   the 2.3 GB archive is streamed, nothing else is kept) -> $VCM_DATA/noise/speech_commands

Usage: python data_prep/download_data.py [ai231] [demand] [speech_noise]     (no argument = all three)
Set HF_TOKEN if Hugging Face rate-limits anonymous downloads. Re-running skips files whose hash is already correct.
"""
from __future__ import annotations

import hashlib, io, sys, tarfile, urllib.request, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcm.common import AI231, DEMAND, REPO, SPEECH_NOISE  # noqa: E402

AI231_REPO, AI231_REV = "airimonda/ai231-me2-voice-commands", "5abbe539a46b9b26ff24e73d2860d1698d44e81f"
DEMAND_REPO, DEMAND_REV = "philgzl/demand", "29f46e613920cc05db198fd4ad7992e874e7f60f"
SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
CHECKSUMS = REPO / "artifacts/checksums"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def expected(name: str) -> dict[str, str]:
    return {line.split()[1]: line.split()[0] for line in (CHECKSUMS / name).read_text().splitlines() if line.strip()}


def ai231() -> None:
    from huggingface_hub import hf_hub_download
    for rel, digest in expected("ai231.sha256").items():
        dest = AI231 / rel
        if dest.is_file() and sha256(dest) == digest:
            print("ok (cached)", rel, flush=True); continue
        hf_hub_download(AI231_REPO, rel, repo_type="dataset", revision=AI231_REV, local_dir=AI231)
        assert sha256(dest) == digest, f"SHA-256 mismatch: {rel}"
        print("ok", rel, flush=True)


def demand() -> None:
    import pyarrow.parquet as pq, soundfile as sf
    from huggingface_hub import snapshot_download
    raw = DEMAND / "parquet"
    snapshot_download(DEMAND_REPO, repo_type="dataset", revision=DEMAND_REV, allow_patterns=["data/16k-*.parquet"], local_dir=raw)
    shards = sorted((raw / "data").glob("16k-*.parquet"))
    assert len(shards) == 6, shards
    for digest_line in (CHECKSUMS / "demand.sha256").read_text().splitlines():
        digest, name = digest_line.split()
        assert sha256(raw / "data" / name) == digest, f"SHA-256 mismatch: {name}"
    n = 0
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=32, columns=["audio", "name"]):
            for payload, name in zip(batch.column(0), batch.column(1)):
                name = name.as_py()  # e.g. DKITCHEN_16k/ch01_00.ogg
                dest = DEMAND / Path(name).with_suffix(".wav")
                if not dest.is_file():
                    audio, sr = sf.read(io.BytesIO(payload.as_py()), dtype="int16")
                    assert sr == 16000 and audio.ndim == 1, (name, sr)
                    dest.parent.mkdir(parents=True, exist_ok=True); sf.write(dest, audio, sr, subtype="PCM_16")
                n += 1
    assert n == 6800, n
    print(f"ok DEMAND: {n} WAVs in {DEMAND}", flush=True)


def speech_noise() -> None:
    want = expected("speech_commands_noise.sha256")
    SPEECH_NOISE.mkdir(parents=True, exist_ok=True)
    if all((SPEECH_NOISE / f).is_file() and sha256(SPEECH_NOISE / f) == d for f, d in want.items()):
        print("ok (cached) Speech Commands background noise", flush=True); return
    with urllib.request.urlopen(SPEECH_COMMANDS_URL) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name)
            if name.parent.name == "_background_noise_" and name.name in want:
                (SPEECH_NOISE / name.name).write_bytes(tar.extractfile(member).read())
                print("extracted", name.name, flush=True)
    for f, d in want.items():
        assert sha256(SPEECH_NOISE / f) == d, f"SHA-256 mismatch: {f}"
        with wave.open(str(SPEECH_NOISE / f)) as w: assert w.getframerate() == 16000
    print("ok Speech Commands background noise", flush=True)


if __name__ == "__main__":
    steps = {"ai231": ai231, "demand": demand, "speech_noise": speech_noise}
    for step in sys.argv[1:] or list(steps):
        steps[step]()
