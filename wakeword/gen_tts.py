"""Step 1 (tts venv): synthesize HELLO ELKIT positives and confusable negatives with Piper (22.05 kHz WAV).

85 speakers: 31 LibriTTS-R (every 30th id), 24 L2-ARCTIC (non-native English), 28 VCTK (UK/IE/Scottish),
lessac (US), alan (GB), Indonesian news voice (Austronesian accent, close to Filipino).
Speakers alternate dev/test by index, so the match threshold is tuned on voices never used for reporting.
"""
import csv, os, sys, wave
from pathlib import Path
from piper import PiperVoice, SynthesisConfig

VOICES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
POS = ["Hello Elkit.", "Hello, Elkit.", "Hello Elkit?", "Okay, hello Elkit.", "Hello Elkit, turn on the lights.",
       "Hello Elkit, set a timer for ten seconds.", "Hello El-kit.", "Hello Elkit!"]
NEG = ["Hello Kitty.", "Hello, kit.", "Hello Elliot.", "Hello, Eli.", "Hello, I'll get it.", "Hello, el kid.",
       "Yellow kit.", "Hello Kate.", "Hello Elk.", "Hello, Ella.", "Hello, Nikki.", "Hello there.", "Hello.",
       "Help, Elliot!", "Hello Eric.", "Fellow kid.", "Hollow kit.", "Hell, I'll get it.", "Hello, skillet.",
       "Hello Elton.", "Below the kit.", "Hello, Elsa.", "Hello, Kit Kat.", "Turn on the lights.",
       "Set a timer for ten seconds.", "What is the weather today?", "Call my mom.", "Marvin.", "Elkhorn.", "Hello, Elkins."]
# GREETING=hi -> HI ELKIT positives + hi-specific confusables, files prefixed "hi_" (merged with the hello set later)
POS_HI = ["Hi Elkit.", "Hi, Elkit.", "Hi Elkit?", "Okay, hi Elkit.", "Hi Elkit, turn on the lights.",
          "Hi Elkit, set a timer for ten seconds.", "Hi El-kit.", "Hi Elkit!"]
NEG_HI = ["Hi, Kitty.", "Hi, kit.", "Hi Elliot.", "Hi, Eli.", "Hi, I'll get it.", "Hi, el kid.", "Hi Kate.", "Hi Elk.",
          "Hi, Ella.", "Hi, Nikki.", "Hi there.", "Hi.", "High kit.", "Hike it.", "Hi Eric.", "Hi, Elton.", "Hi, Elsa.",
          "Hi, Kit Kat.", "Hi, Elkins.", "Hide it.", "Hi, Nelly.", "Bye, Elliot.", "My kit.", "Hi, skillet."]
PFX = ""
if os.environ.get("GREETING") == "hi": POS, NEG, PFX = POS_HI, NEG_HI, "hi_"
OUT.mkdir(parents=True, exist_ok=True)
lib = PiperVoice.load(str(VOICES / "en_US-libritts_r-medium.onnx"))
speakers = [("libritts", lib, s) for s in range(0, 904, 30)]
l2 = PiperVoice.load(str(VOICES / "en_US-l2arctic-medium.onnx"))  # 24 non-native English speakers
speakers += [("l2arctic", l2, s) for s in range(24)]
vctk = PiperVoice.load(str(VOICES / "en_GB-vctk-medium.onnx"))  # UK / Irish / Scottish accents
speakers += [("vctk", vctk, s) for s in range(0, 109, 4)]
speakers += [("lessac", PiperVoice.load(str(VOICES / "en_US-lessac-medium.onnx")), None),
             ("alan", PiperVoice.load(str(VOICES / "en_GB-alan-medium.onnx")), None),
             ("indonesian", PiperVoice.load(str(VOICES / "id_ID-news_tts-medium.onnx")), None)]
rows = []
for si, (name, voice, sid) in enumerate(speakers):
    split = "dev" if si % 2 == 0 else "test"
    spk = f"{name}{'' if sid is None else sid}"
    for label, texts in (("pos", POS), ("neg", NEG)):
        for ti, text in enumerate(texts):
            if label == "neg" and (ti + si) % 3:  # each negative on a third of the speakers (~630 total)
                continue
            ls = (0.85, 1.0, 1.2)[(ti + si) % 3]
            path = OUT / f"{PFX}{label}_{spk}_{ti:02d}.wav"
            with wave.open(str(path), "wb") as w:
                voice.synthesize_wav(text, w, syn_config=SynthesisConfig(speaker_id=sid, length_scale=ls))
            rows.append(dict(file=path.name, label=label, text=text, speaker=spk, split=split, length_scale=ls))
with open(OUT / "tts_index.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, rows[0].keys()); wr.writeheader(); wr.writerows(rows)
print(len(rows), "clips")
