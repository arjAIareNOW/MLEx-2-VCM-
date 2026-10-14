"""Step 1b (tts venv + transformers): Filipino voices from Meta MMS-TTS (VITS, CC-BY-NC-4.0, 16 kHz).

8 Philippine-language voices: Tagalog, Cebuano, Ilocano, Hiligaynon, Waray, Kapampangan, Bikol, Maranao.
Each is one speaker; 4 seeds x speaking rates give natural variation. Text is in Filipino reading
(Taglish commands included), so "hello elkit" is pronounced the way a Filipino would read it.
Voices alternate dev/test.
"""
import csv, os, sys
from pathlib import Path
import numpy as np, soundfile as sf, torch
from transformers import VitsModel, AutoTokenizer

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LANGS = ["tgl", "ceb", "ilo", "hil", "war", "pam", "bcl", "mrw"]
POS = ["hello elkit", "okay hello elkit", "hello elkit buksan mo ang ilaw", "hello elkit turn on the lights",
       "hello elkit set a timer for ten seconds", "hello elkit patayin ang ilaw", "hello elkit ano ang oras", "helo elkit"]
NEG = ["hello kit", "hello ate", "hello kuya", "hello elsa", "hello eli", "kumusta ka", "buksan mo ang ilaw",
       "salamat po", "hello kitty", "elektrik", "marvin", "turn on the lights", "hello", "ilaw sa kusina", "sige na",
       "ano ang oras", "hello pakit", "yelo kit", "hello el", "hello kid"]
# GREETING=hi -> HI ELKIT ("hay" = how a Filipino spells /hai/), files prefixed "hi_"
POS_HI = ["hi elkit", "okay hi elkit", "hi elkit buksan mo ang ilaw", "hi elkit turn on the lights",
          "hi elkit set a timer for ten seconds", "hi elkit patayin ang ilaw", "hi elkit ano ang oras", "hay elkit"]
NEG_HI = ["hi kit", "hi ate", "hi kuya", "hi elsa", "hi eli", "hi", "hi kitty", "hi el", "hi kid", "hay naku",
          "hi elliot", "hay kit", "high kit", "hi kate", "kumusta ka", "salamat po", "turn on the lights", "ano ang oras"]
PFX = ""
if os.environ.get("GREETING") == "hi": POS, NEG, PFX = POS_HI, NEG_HI, "hi_"
rows = []
for li, lang in enumerate(LANGS):
    tok = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{lang}"); model = VitsModel.from_pretrained(f"facebook/mms-tts-{lang}").eval()
    split = "dev" if li % 2 == 0 else "test"
    for label, texts, seeds in (("pos", POS, 4), ("neg", NEG, 2)):
        for ti, text in enumerate(texts):
            for s in range(seeds):
                torch.manual_seed(1000 * li + 10 * ti + s)
                model.speaking_rate = (0.85, 1.0, 1.15, 0.95)[s]
                with torch.no_grad():
                    y = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
                name = f"{PFX}{label}_mms{lang}_{ti:02d}_{s}.wav"
                sf.write(OUT / name, y / max(1e-4, np.abs(y).max()) * 0.9, model.config.sampling_rate, subtype="PCM_16")
                rows.append(dict(file=name, label=label, text=text, speaker=f"mms_{lang}", split=split, length_scale=model.speaking_rate))
    print(lang, "done", flush=True)
with open(OUT / "tts_index.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, rows[0].keys()); wr.writeheader(); wr.writerows(rows)
print(len(rows), "clips")
