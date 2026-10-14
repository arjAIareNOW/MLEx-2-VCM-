"""Download what the Elkit wake-word pipeline needs (on top of data_prep/download_data.py), pinned and checked.

  piper     38 Piper TTS voices (rhasspy/piper-voices @ c10ece1a, MIT; per-voice licences in each MODEL_CARD)
                                                          -> $VCM_DATA/tts_voices/*.onnx(.json)
  mms       Meta MMS-TTS VITS models for 25 languages (facebook/mms-tts-*, CC-BY-NC-4.0: non-commercial)
                                                          -> Hugging Face cache (revisions in artifacts/checksums)
  mls       Multilingual LibriSpeech Spanish 9_hours / dev / test (facebook/multilingual_librispeech @ 2e83e618,
            CC-BY-4.0); real Spanish "el" is cut from it  -> $VCM_DATA/mls_es/
  mswc      Multilingual Spoken Words Corpus English split lists (MLCommons/ml_spoken_words @ 0bc9df68, CC-BY-4.0);
            the clips themselves are streamed by wakeword/fetch_mswc.py -> $VCM_DATA/mswc/
  speech_commands  Speech Commands v0.02 word clips (CC-BY-4.0, 2.3 GB) -> $VCM_DATA/speech_commands/

Usage: python wakeword/download_wakeword_data.py [piper] [mms] [mls] [mswc] [speech_commands]   (none = all)
"""
import hashlib, os, shutil, sys, tarfile, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("VCM_DATA", REPO / "data")).expanduser()
CHECK = REPO / "artifacts/checksums"
PIPER_REV = "c10ece1aade47bb51c153c893d14e5bf8e5b7117"
MLS_REV = "2e83e61823b4c47dcbcb1980bb88601274127609"
MSWC_REV = "0bc9df68e92fd6bb54176bf7eb29e2b9e97cb218"
SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def table(name):
    return [l.split() for l in (CHECK / name).read_text().splitlines() if l.strip()]


def piper():
    from huggingface_hub import hf_hub_download
    out = DATA / "tts_voices"; out.mkdir(parents=True, exist_ok=True)
    for digest, fname in table("piper_voices.sha256"):
        voice = fname[:-5]                      # e.g. en_US-lessac-medium
        lang_region, name, quality = voice.split("-", 2)[0], voice.split("-")[1], voice.split("-")[-1]
        folder = f"{lang_region.split('_')[0]}/{lang_region}/{name}/{quality}"
        for f in (fname, fname + ".json"):
            if (out / f).is_file() and (f.endswith(".json") or sha256(out / f) == digest): continue
            p = hf_hub_download("rhasspy/piper-voices", f"{folder}/{f}", revision=PIPER_REV)
            shutil.copy(p, out / f)
        assert sha256(out / fname) == digest, fname
        print("ok", voice, flush=True)


def mms():
    from huggingface_hub import snapshot_download
    for rev, repo in table("mms_tts_revisions.txt"):
        snapshot_download(repo, revision=rev); print("ok", repo, flush=True)


def mls():
    from huggingface_hub import hf_hub_download
    out = DATA / "mls_es"; out.mkdir(parents=True, exist_ok=True)
    for digest, fname in table("mls_spanish.sha256"):  # 9_hours.parquet, dev.parquet, test.parquet
        if not ((out / fname).is_file() and sha256(out / fname) == digest):
            p = hf_hub_download("facebook/multilingual_librispeech", f"spanish/{fname[:-8]}-00000-of-00001.parquet",
                                repo_type="dataset", revision=MLS_REV)
            shutil.copy(p, out / fname)
        assert sha256(out / fname) == digest, fname
        print("ok mls", fname, flush=True)


def mswc():
    from huggingface_hub import hf_hub_download
    out = DATA / "mswc"; out.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download("MLCommons/ml_spoken_words", "data/splits/en/splits.tar.gz", repo_type="dataset", revision=MSWC_REV)
    assert sha256(p) == table("mswc_en_splits.sha256")[0][0]
    with tarfile.open(p) as t:
        for m in t.getmembers():
            if m.isfile() and Path(m.name).name in ("train.csv", "dev.csv", "test.csv", "version.txt"):
                (out / Path(m.name).name).write_bytes(t.extractfile(m).read())
    print("ok mswc split lists; now run: python wakeword/fetch_mswc.py && EXTRA_WORDS=high python wakeword/fetch_mswc.py", flush=True)


def speech_commands():
    out = DATA / "speech_commands"; out.mkdir(parents=True, exist_ok=True)
    if len(list(out.glob("*/*.wav"))) > 100000:
        print("ok (cached) speech_commands"); return
    with urllib.request.urlopen(SPEECH_COMMANDS_URL) as r, tarfile.open(fileobj=r, mode="r|gz") as t:
        t.extractall(out)
    print("ok speech_commands", len(list(out.glob("*/*.wav"))), "clips", flush=True)


if __name__ == "__main__":
    steps = {"piper": piper, "mms": mms, "mls": mls, "mswc": mswc, "speech_commands": speech_commands}
    for s in sys.argv[1:] or list(steps):
        steps[s]()
