"""Stream MSWC English opus shards from Hugging Face and keep only selected clips (no full download on disk).

Kept: every clip (capped per word) of WORDS = wake-word parts + real-speech confusables for ELKORIS / HELLO ELKIT,
plus a seeded random sample of other words (real-speech negatives for false-wakes estimates).
MSWC = Multilingual Spoken Words Corpus (MLCommons, CC-BY 4.0), 1 s word clips cut from Common Voice.
Writes clips/<word>/<file>.opus and selected.csv (LINK, WORD, SPLIT, SPEAKER, GENDER, KIND).
"""
import csv, os, random, sys, tarfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

D = Path(os.environ.get("VCM_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser() / "mswc"  # split lists from download_wakeword_data.py mswc
URL = "https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/main/data/opus/en/{split}/audio/{i}.tar.gz"
SHARDS = {"train": 88, "dev": 12, "test": 12}
WORDS = """hello elk elks kit kits kid kids hell well yellow kitty kitten kate elliot eli ella elle skit halo hollow fellow
jello cello below else elf elm elbow eleven elephant elegant elevator elect electric kick kiss kitchen help hey okay
chorus course chris kris boris iris doris morris gloria glorious victoria elvis dolores calories horse coral corridor
korea korean carlos escort elixir marvin lori florist elkhart""".split()
CAP, N_RANDOM = 3000, 30000
SEL = "selected.csv"
if os.environ.get("EXTRA_WORDS"):  # second pass for words added later (e.g. "high" = "hi" for HI ELKIT): no random sample
    WORDS, N_RANDOM, SEL = os.environ["EXTRA_WORDS"].split(), 0, "selected_extra.csv"

rows = [dict(r, SPLIT=sp) for sp in SHARDS for r in csv.DictReader(open(D / f"{sp}.csv"))]
want = {}
by_word = {}
for r in rows: by_word.setdefault(r["WORD"], []).append(r)
rng = random.Random(23101)
for w in WORDS:
    rs = by_word.get(w, []); rng.shuffle(rs)
    for r in rs[:CAP]: want[r["LINK"]] = dict(r, KIND="target")
others = [r for r in rows if r["WORD"] not in WORDS]
for r in (rng.sample(others, N_RANDOM) if N_RANDOM else []): want[r["LINK"]] = dict(r, KIND="random")
print(len(want), "clips wanted", flush=True)
keys = {l.replace("/", "_"): l for l in want}  # tar members are flattened word_common_voice_en_N.opus


def shard(job):
    split, i = job
    for attempt in range(3):
        try:
            got = 0
            with urllib.request.urlopen(URL.format(split=split, i=i), timeout=120) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tf:
                for m in tf:
                    name = m.name.lstrip("/")
                    link = keys.get(name) or keys.get(name.replace("/", "_"))
                    if link and m.isfile():
                        out = D / "clips" / link; out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(tf.extractfile(m).read()); got += 1
            print(f"{split}/{i} kept {got}", flush=True); return got
        except Exception as e:
            print(f"{split}/{i} attempt {attempt} failed: {e}", flush=True); time.sleep(30)
    return 0


jobs = [(s, i) for s, n in SHARDS.items() for i in range(n)]
if len(sys.argv) > 1: jobs = jobs[:int(sys.argv[1])]
with ThreadPoolExecutor(4) as ex:
    total = sum(ex.map(shard, jobs))
with open(D / SEL, "w", newline="") as f:
    wr = csv.DictWriter(f, ["LINK", "WORD", "SPLIT", "SPEAKER", "GENDER", "KIND"], extrasaction="ignore"); wr.writeheader()
    wr.writerows(r for l, r in want.items() if (D / "clips" / l).exists())
print("done, kept", total, flush=True)
