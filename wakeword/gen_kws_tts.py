"""KWS step 1 (tts venv): TRAINING audio for the dedicated HELLO ELKIT spotter, speaker-disjoint from the eval set.

Eval (gen_tts.py / gen_mms.py) uses LibriTTS-R ids % 30 == 0, VCTK ids % 4 == 0, all 24 L2-ARCTIC, lessac, alan,
Indonesian (Piper) and 8 MMS languages. Here: other LibriTTS-R / VCTK ids, 17 more English Piper voices (incl. 18 ARCTIC
+ 12 ARU + 4 SEMAINE speakers), 12 non-English Piper voices reading English (accented; nl_NL 52 + ne_NP 18 speakers),
12 OTHER Philippine-language MMS voices + extra MMS languages (whichever are cached).
Confusable negatives: a large near-miss list, AUTO-FILTERED against the eval negatives (gen_tts.py / gen_mms.py NEG):
a candidate is dropped if it contains an eval content word, or a word / whole phrase that phonemizes (espeak en-us)
identically to one. The dropped list is written to OUT/dropped_negatives.txt.
Kinds: pos = the wake word alone (so its position is known), lead = "hey/okay..." alone, cmd = command alone,
neg = confusables / other speech. The trainer concatenates lead + pos + cmd on the fly.
~10 % of speakers -> split=val (checkpoint selection only).
"""
import ast, csv, os, re, sys, wave, zlib
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
VOICES, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
POS = ["Hello Elkit.", "Hello Elkit?", "Hello Elkit!", "Hello, Elkit,", "Hello El-kit.", "Hello Elkitt."]
LEAD = ["Okay.", "Oh.", "Um.", "Hey.", "Hi."]
CMD = ["Turn on the lights.", "Set an alarm for seven.", "What time is it?", "Play some music.", "Stop.",
       "Increase the volume.", "Close the curtains.", "Start the timer."]
NEG_CANDIDATES = [
    # the frame word alone / with other words (must NOT wake)
    "Hello.", "Hello, hello.", "Hello everyone.", "Hello world.", "Hello, how are you?", "Hello again.", "Hallo.", "Hullo.",
    "Hello, good morning.", "Hello, can you hear me?", "Hello? Anyone there?",
    # hello + near-miss names / words
    "Hello Mickey.", "Hello Kevin.", "Hello Keith.", "Hello Kim.", "Hello Elizabeth.", "Hello Ellen.", "Hello Elvis.",
    "Hello Elmo.", "Hello Ezekiel.", "Hello kitchen.", "Hello, gift.", "Hello, quick.", "Hello Alex.", "Hello Felix.",
    "Hello Skip.", "Hello Kip.", "Hello Kat.", "Hello, kick.", "Hello Elijah.", "Hello, elf.", "Hello, elm.",
    "Hello, elbow.", "Hello Ellie.", "Hello Elena.", "Hello Elias.", "Hello, Ellis.", "Hello Kristen.", "Hello, Kilt.",
    "Hello, Cal.", "Hello Kelly.", "Hello Kiki.", "Hello, Hellcat.", "Hello, Elkhart.", "Hello, Belgium.", "Hello Melissa.",
    "Hello, Vel.", "Hello, L.", "Hello, Elk River.", "Hello, Kitten.", "Hello, Kitsap.", "Hello Kiddo.", "Hello, skit.",
    "Hello, Electric.", "Hello, Elegant.", "Hello, Eleven.", "Hello, Elevator.", "Hello, Elephant.",
    # sound-alikes without hello
    "Jello kit.", "Cello kit.", "Mellow kit.", "Yellow kitten.", "Hello kits.", "Elk it.", "L kit.", "Help a kid.",
    "Tell a kid.", "Spell it.", "Yell it.", "Sell kits.", "Hell's kitchen.", "Shell kit.", "Well, get it.", "Hold it.",
    "Hollow kid.", "Below it.", "Hello, Kit Carson.", "Halo kit.", "Hello Kit Harington.", "Fellow kit.", "Bellow it.",
    "Follow it.", "Swallow it.", "Hello, gal kit.", "Elliot kit.", "Hey, Elkins.", "Oh, Elkit is late.",
    "Excellent.", "I love it.", "Turn it off.", "Not now.", "Thank you.", "Good night.", "See you later.",
]
MMS_LANGS = "pag krj mbt sml agn ify ifk ifu ifb itv kne mbb".split()
MMS_EXTRA = "spa por zlm ind jav sun kor hin ara cmn tha swh yor tur fra deu vie urd ben".split()  # used if cached + Latin
MMS_POS = ["hello elkit", "hello elkit?", "hello elkit!", "hello el kit", "helo elkit"]
MMS_LEAD = ["hoy", "okay", "uy", "o"]
MMS_CMD = ["buksan mo ang ilaw", "patayin ang ilaw", "ano ang oras", "turn on the lights", "salamat"]
MMS_NEG_CANDIDATES = [
    "hello", "helo po", "hello hello", "hello kumusta", "hello ate", "hello kuya", "hello beh", "hello mars", "hello pare",
    "hello elena", "hello elias", "hello kim", "hello kiko", "hello kikay", "hello kilig", "hello kita", "hello kitang",
    "hello pakita", "hello elektrisidad", "hello kalye", "yelo kita", "hello sige", "hello salamat", "elk", "kit kat",
    "hello elmer", "hello elvie", "hello elsie", "hello kapitan", "hello kilos", "hello kitkat", "halo halo",
    "kumusta ka", "salamat po", "sige na", "ilaw sa kusina", "tara na", "ay naku", "kumain ka na", "hello kris"]
# GREETING=hi -> HI ELKIT training audio (lead / cmd come from the hello run), files prefixed "hi_"
POS_HI = ["Hi Elkit.", "Hi Elkit?", "Hi Elkit!", "Hi, Elkit,", "Hi El-kit.", "Hi Elkitt."]
NEG_CANDIDATES_HI = [
    "Hi.", "Hi, hi.", "Hi everyone.", "Hi, how are you?", "Hi again.", "Hi, good morning.", "Hey there.", "Hiya.",
    "Hi Mickey.", "Hi Kevin.", "Hi Keith.", "Hi Kim.", "Hi Ellen.", "Hi Elvis.", "Hi Elmo.", "Hi, kitchen.", "Hi, gift.",
    "Hi Alex.", "Hi Felix.", "Hi Skip.", "Hi Kip.", "Hi Kat.", "Hi, kick.", "Hi Elijah.", "Hi, elf.", "Hi Ellie.",
    "Hi Elena.", "Hi Kelly.", "Hi Kiki.", "Hi, Elkhart.", "Hi Melissa.", "Hi, Kitten.", "Hi Kiddo.", "Hi, skit.",
    "Hi, Electric.", "Hi, Eleven.", "High kick.", "Hike up.", "Hi-fi kit.", "Wi-Fi kit.", "Sci-fi kit.", "Buy a kit.",
    "Tie it.", "Pie kit.", "Why, Elliot?", "I'll get it.", "Hi, Elk River.", "Hi, gal kit.", "Hi, Kit Carson.",
    "Hi, Ezekiel.", "Hi, Elmer.", "Height kit.", "Hi, help a kid."]
MMS_POS_HI = ["hi elkit", "hi elkit?", "hi elkit!", "hi el kit", "hay elkit"]
MMS_NEG_CANDIDATES_HI = [
    "hi po", "hi hi", "hi kumusta", "hi beh", "hi mars", "hi pare", "hi elena", "hi kim", "hi kiko", "hi kikay",
    "hi kita", "hi elmer", "hi elvie", "hi kapitan", "hi kitkat", "hay buhay", "hi kris", "hi kilig", "hay salamat"]
PFX = ""
if os.environ.get("GREETING") == "hi":
    POS, MMS_POS, NEG_CANDIDATES, MMS_NEG_CANDIDATES, PFX = POS_HI, MMS_POS_HI, NEG_CANDIDATES_HI, MMS_NEG_CANDIDATES_HI, "hi_"
    LEAD, CMD, MMS_LEAD, MMS_CMD = ["Okay.", "Oh.", "Um.", "Hello."], [], ["hoy", "okay"], []


def _list_from(path, name="NEG"):
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def words(s):
    return re.findall(r"[a-z']+", s.lower().replace("-", " "))


def filter_negatives(cands, log):
    """drop candidates that share a content word, a homophone word, or the whole-phrase sound with an eval negative."""
    from piper.phonemize_espeak import EspeakPhonemizer
    ph = EspeakPhonemizer()
    sound = lambda s: "".join("".join(p) for p in ph.phonemize("en-us", s)).replace(" ", "").replace("ˈ", "").replace("ˌ", "").replace("ː", "")
    ev = [t for f in ("gen_tts.py", "gen_mms.py") for n in ("NEG", "NEG_HI") for t in _list_from(HERE / f, n)]
    stop = {w for t in LEAD + CMD + MMS_LEAD + MMS_CMD for w in words(t)}
    stop |= {"hi", "hay", "help", "hello", "helo", "kit", "el", "there", "get", "it", "call", "is", "a", "the", "my", "for", "of", "ka", "na", "po", "sa", "mo", "ang", "here", "bad"}
    ev_words = {w for t in ev for w in words(t)} - stop
    ev_word_snd = {sound(w): w for w in ev_words}
    ev_phrase_snd = {sound(t): t for t in ev}
    keep = []
    for c in dict.fromkeys(cands):
        cw = set(words(c))
        if not cw - stop: keep.append(c); continue  # frame words only (e.g. "Hello."): the core negative, kept on purpose
        why = (f"eval word {sorted(cw & ev_words)}" if cw & ev_words else
               next((f"homophone {w}~{ev_word_snd[sound(w)]}" for w in cw - stop if sound(w) in ev_word_snd), None) or
               (f"phrase sounds like {ev_phrase_snd[sound(c)]!r}" if sound(c) in ev_phrase_snd else None))
        if why: log.append(f"{c}\t{why}")
        else: keep.append(c)
    return keep


def split_of(spk):
    return "val" if zlib.crc32(spk.encode()) % 10 == 0 else "train"


def piper_job(job):
    from piper import PiperVoice, SynthesisConfig
    vfile, sid, spk, neg_keep = job
    voice = PiperVoice.load(str(VOICES / vfile))
    rows = []
    for kind, texts, rates in (("pos", POS, (0.85, 1.0, 1.2)), ("lead", LEAD, (1.0,)), ("cmd", CMD, (1.0,)), ("neg", NEG, (0.9, 1.15))):
        for ti, text in enumerate(texts):
            for ri, ls in enumerate(rates):
                if kind == "neg" and (ti * 7 + ri * 3 + zlib.crc32(spk.encode())) % neg_keep: continue  # each negative on 1/neg_keep of speaker-rates
                name = f"{PFX}{kind}_{spk}_{ti:03d}_{ri}.wav"
                if not (OUT / name).exists() or (OUT / name).stat().st_size < 1000:  # resume after a crash
                    with wave.open(str(OUT / name), "wb") as w:
                        voice.synthesize_wav(text, w, syn_config=SynthesisConfig(speaker_id=sid, length_scale=ls))
                rows.append(dict(file=name, kind=kind, text=text, speaker=spk, split=split_of(spk)))
    return rows


def mms_job(lang):
    import numpy as np, soundfile as sf, torch
    from transformers import VitsModel, AutoTokenizer
    torch.set_num_threads(1)
    tok = AutoTokenizer.from_pretrained(f"facebook/mms-tts-{lang}"); model = VitsModel.from_pretrained(f"facebook/mms-tts-{lang}").eval()
    spk, rows = f"mms_{lang}", []
    for kind, texts, seeds in (("pos", MMS_POS, 8), ("lead", MMS_LEAD, 2), ("cmd", MMS_CMD, 2), ("neg", MMS_NEG, 2)):
        for ti, text in enumerate(texts):
            for s in range(seeds):
                torch.manual_seed(7000 + 100 * ti + s); model.speaking_rate = (0.8, 0.9, 1.0, 1.1, 1.2)[s % 5]
                ids = tok(text, return_tensors="pt")
                if ids["input_ids"].shape[1] < 3: break  # letters missing from this language's vocab (e.g. "o" in mbt)
                name = f"{PFX}{kind}_{spk}_{ti:03d}_{s}.wav"
                with torch.no_grad():
                    y = model(**ids).waveform[0].numpy()
                sf.write(OUT / name, y / max(1e-4, np.abs(y).max()) * 0.9, 16000, subtype="PCM_16")
                rows.append(dict(file=name, kind=kind, text=text, speaker=spk, split=split_of(spk)))
    return rows


def mms_usable(lang):
    """cached offline and its character vocab covers the Latin wake word."""
    import json
    from huggingface_hub import try_to_load_from_cache
    paths = [try_to_load_from_cache(f"facebook/mms-tts-{lang}", f) for f in ("model.safetensors", "vocab.json")]
    if not all(isinstance(p, str) for p in paths): return False
    return set("helokit") <= set(json.load(open(paths[1])))


log = []
NEG = filter_negatives(NEG_CANDIDATES, log)
MMS_NEG = filter_negatives(MMS_NEG_CANDIDATES, log)

if __name__ == "__main__":
    (OUT / "dropped_negatives.txt").write_text("\n".join(log) + "\n")
    print(f"negatives kept: piper {len(NEG)}/{len(NEG_CANDIDATES)}, mms {len(MMS_NEG)}/{len(MMS_NEG_CANDIDATES)}", flush=True)
    jobs = [("en_US-libritts_r-medium.onnx", s, f"libritts{s}", 4) for s in range(7, 904, 15) if s % 30]
    jobs += [("en_GB-vctk-medium.onnx", s, f"vctk{s}", 4) for s in range(1, 109, 3) if s % 4]
    jobs += [("en_US-arctic-medium.onnx", s, f"arctic{s}", 4) for s in range(18)]
    jobs += [("en_GB-aru-medium.onnx", s, f"aru{s}", 4) for s in range(12)]
    jobs += [("en_GB-semaine-medium.onnx", s, f"semaine{s}", 4) for s in range(4)]
    jobs += [(f"{v}.onnx", None, v.split("-")[1], 2) for v in
             ["en_US-amy-medium", "en_US-joe-medium", "en_US-kusal-medium", "en_US-john-medium", "en_US-bryce-medium",
              "en_US-ryan-medium", "en_US-hfc_female-medium", "en_US-hfc_male-medium", "en_GB-jenny_dioco-medium",
              "en_GB-southern_english_female-low", "en_GB-alba-medium",
              "en_GB-cori-medium", "en_GB-northern_english_male-medium", "en_US-kristin-medium", "en_US-ljspeech-medium",
              "en_US-norman-medium", "en_US-sam-medium"]]
    # non-English voices reading the English text = accented English (espeak phonemizes it with their language rules)
    jobs += [(f"{v}.onnx", None, v.split("-")[0].replace("_", "") + v.split("-")[1], 2) for v in
             ["es_ES-davefx-medium", "es_MX-ald-medium", "pt_BR-faber-medium", "it_IT-riccardo-x_low", "sw_CD-lanfrica-medium",
              "tr_TR-dfki-medium", "fr_FR-siwis-medium", "pl_PL-gosia-medium", "ro_RO-mihai-medium", "ar_JO-kareem-medium"]]
    jobs += [("nl_NL-mls-medium.onnx", s, f"nlmls{s}", 4) for s in range(52)]
    jobs += [("ne_NP-google-medium.onnx", s, f"negoogle{s}", 4) for s in range(18)]
    langs = MMS_LANGS + [l for l in MMS_EXTRA if mms_usable(l)]
    print("mms languages:", " ".join(langs), flush=True)
    if os.environ.get("SMOKE"):
        pick = [j for j in jobs if j[2] in ("libritts7", "cori", "esESdavefx", "arJOkareem", "nlmls0", "negoogle0")]
        jobs, langs = pick, [MMS_LANGS[0]] + [l for l in langs if l in MMS_EXTRA][:1]
    with Pool(int(os.environ.get("WORKERS", "8"))) as p:
        rows = [r for out in p.imap_unordered(piper_job, jobs) for r in out]
        print(len(rows), "piper clips", flush=True)
        rows += [r for out in p.imap_unordered(mms_job, langs) for r in out]
    with open(OUT / "index_tts.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    print(len(rows), "clips total", flush=True)
