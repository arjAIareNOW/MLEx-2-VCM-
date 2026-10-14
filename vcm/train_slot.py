"""Train the slot-value model: 18 outputs = 3 schema values for each of the 6 slotted intents (TIMER, ALARM,
TEMPERATURE, BRIGHTNESS, COLOR, ... see SLOT_CLASSES). At inference only the 3 values of the predicted intent compete.

Recipe (docs/05_slot_model.md):
  data      ai231_filtered_lenient train + supplemental_fil slotted clips (speaker-disjoint validation, seed 23101,
            artifacts/splits/slot_validation_speakers.json) + 7,000 real-human ai231 `numerals` clips (700 per number
            slot, data_prep/numerals_subset.py)
  augment   gain / speed / shift / noise (vcm.augment.WaveformAugmentor) with Speech Commands noise + 13 DEMAND rooms
            (equal weight); p=0.5 placement inside a 5 s window; p=0.2 tail crop starting just before the slot word
            (Whisper word timings, artifacts/whisper/slot_word_timestamps.jsonl.gz); 2 frequency masks
  optimiser AdamW lr 2e-3 / wd 0.05, warm-up 5 epochs + cosine, batch 128, 150 epochs, label smoothing 0.1, bf16, EMA
  selection validation score = mean(accuracy on real-human clips, accuracy on synthetic clips), better of raw/EMA,
            best epoch = centre of the best 5-epoch moving average (no early stopping; every epoch checkpointed)
  test      test + holdout, clean and with 4 DEMAND rooms never used in training at 5 / 10 / 20 dB SNR

Usage: python -m vcm.train_slot --output runs/slot [--smoke]
"""
from __future__ import annotations

import argparse, csv, gzip, json, os, re, shutil, time, wave, zlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset

from .augment import WaveformAugmentor, load_noise_dir, spec_augment
from .common import AI231, ARTIFACTS, CLASSES, DATA_ROOT, DEMAND, SPEECH_NOISE, collect_environment, sha256_file
from .data import check_validation_speakers, eligible, read
from .features import LogMel, decode_wav
from .models import SlotTCDSCNN
from .train_intent import lr_at

WINDOW = 80_000
DEMAND_HELDOUT = ("DLIVING", "OMEETING", "PCAFETER", "STRAFFIC")  # evaluation only, never mixed into training
NOISE_SNRS = (5, 10, 20)
NOISE_CONDS = tuple(f"demand_{s}db" for s in NOISE_SNRS)
NUMWORD = {"one": "1", "six": "6", "eight": "8", "nine": "9", "ten": "10", "thirty": "30", "eighteen": "18",
           "twenty": "20", "sixty": "60", "hundred": "100"}


def slot_schema(ai231: Path):
    var = list(csv.DictReader((ai231 / "variations.csv").open()))
    slotted = list(dict.fromkeys(r["label"] for r in var if r["value"]))
    classes = list(dict.fromkeys(f"{r['label']}|{r['value']}" for r in var if r["value"]))
    groups = torch.tensor([[classes.index(f"{c}|{v}") for v in dict.fromkeys(r["value"] for r in var if r["label"] == c)]
                           for c in slotted])  # [6, 3]
    return slotted, classes, groups


def load_demand(heldout: bool) -> dict[str, list[torch.Tensor]]:
    """environment -> ch01 chunks; held-out or training environments."""
    out = {}
    for d in sorted(DEMAND.glob("*_16k")):
        env = d.name[:-4]
        if (env in DEMAND_HELDOUT) != heldout: continue
        chunks = []
        for f in sorted(d.glob("ch01_*.wav")):
            with wave.open(str(f), "rb") as h:
                assert (h.getframerate(), h.getnchannels(), h.getsampwidth()) == (16000, 1, 2)
                chunks.append(torch.from_numpy(np.frombuffer(h.readframes(h.getnframes()), dtype="<i2").copy()).float().div_(32768.0))
        if chunks: out[env] = chunks
    return out


def noise_env(key: str, envs: list[str]) -> str:
    return envs[zlib.crc32(key.encode()) % len(envs)]


def mix_fixed_noise(x, key, noises, snr_db):
    """Deterministic per clip: environment, chunk and offset depend only on the clip key."""
    envs = sorted(noises); h = zlib.crc32(key.encode()); chunks = noises[noise_env(key, envs)]
    g = torch.Generator().manual_seed(h)
    seg = chunks[int(torch.randint(len(chunks), (), generator=g))]
    if len(seg) < x.shape[-1]: seg = seg.repeat((x.shape[-1] + len(seg) - 1) // len(seg))
    start = int(torch.randint(len(seg) - x.shape[-1] + 1, (), generator=g)); seg = seg[start:start + x.shape[-1]][None]
    scale = x.square().mean().sqrt().clamp_min(1e-5) / (seg.square().mean().sqrt().clamp_min(1e-5) * 10 ** (snr_db / 20))
    return (x + seg * scale).clamp(-1, 1)


def tok(w: str) -> str:
    t = re.sub(r"[^a-z0-9]", "", w.lower()); t = re.sub(r"(am|pm|degrees?|percent)$", "", t)
    return NUMWORD.get(t, t)


def load_timings(path: Path) -> dict[str, tuple[float, float]]:
    """clip key -> (start, end) seconds of the slot word, from Whisper word timestamps."""
    out = {}
    for line in gzip.open(path, "rt"):
        r = json.loads(line); key = f"{r['split']}|{r['file']}"
        core = tok((r["slot_value"] or "").split()[0].split(":")[0]) if r["slot_value"] else ""
        hit = [w for w in r["words"] if tok(w["w"]) == core or tok(w["w"]).startswith(core) and core.isalpha()]
        if hit: out[key] = (hit[0]["s"], hit[0]["e"])
    return out


class SlotDataset(Dataset):
    def __init__(self, items, n_out, sources, train=False, augmentor=None, timings=None, fixed_noise=None):
        self.items, self.n_out, self.sources, self.train = items, n_out, sources, train
        self.augmentor, self.timings, self.fixed_noise = augmentor, timings or {}, fixed_noise
        self.front_end = LogMel()

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        src, row, label, key, human, group = self.items[i]
        target = torch.zeros(self.n_out); target[label] = 1.0
        x = decode_wav(self.sources[src][row].as_py()["bytes"])[..., :WINDOW]
        if self.fixed_noise is not None: x = mix_fixed_noise(x, key, *self.fixed_noise)
        if self.train and key in self.timings:
            s, _ = (int(v * 16000) for v in self.timings[key])
            if float(torch.rand(())) < 0.2:  # tail crop starting just before the slot word
                start = max(0, s - int(16000 * (0.05 + 0.25 * float(torch.rand(())))))
                if x.shape[-1] - start > 1600: x = x[..., start:]
        if self.train:
            x = self.augmentor(x)
            if float(torch.rand(())) < 0.5 and x.shape[-1] < WINDOW:  # place inside a 5-s window, as on test / the Pi
                buf = torch.randn(1, WINDOW) * 3e-4; off = int(torch.randint(WINDOW - x.shape[-1] + 1, ()))
                buf[..., off:off + x.shape[-1]] += x; x = buf.clamp(-1, 1)
        return self.front_end(x), target, group, human


def collate(batch):
    feats, targets, groups, human = zip(*batch)
    lengths = torch.tensor([f.shape[-1] for f in feats])
    x = pad_sequence([f.squeeze(0).transpose(0, 1) for f in feats], batch_first=True).transpose(1, 2).unsqueeze(1)
    return x, lengths, torch.stack(targets), torch.tensor(groups), torch.tensor(human)


def soft_ce(logits, target, smoothing):
    t = target * (1 - smoothing) + smoothing / target.shape[1]
    return -(t * logits.log_softmax(1)).sum(1).mean()


def restrict(groups, logits, target, group):
    idx = groups.to(logits.device)[group]
    return logits.gather(1, idx), target.gather(1, idx), idx


@torch.inference_mode()
def predict(model, loader, groups, device):
    model.eval(); ys, ps, hs = [], [], []
    for x, lengths, target, group, human in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x.to(device), lengths.to(device)).float()
        lg, tg, idx = restrict(groups, logits, target.to(device), group.to(device))
        p = idx.gather(1, lg.argmax(1)[:, None])[:, 0]; y = idx.gather(1, tg.argmax(1)[:, None])[:, 0]
        ys += y.tolist(); ps += p.tolist(); hs += human.tolist()
    return np.array(ys), np.array(ps), np.array(hs)


@torch.inference_mode()
def mean_loss(model, loader, groups, device):
    model.eval(); tot = n = 0
    for x, lengths, target, group, _ in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x.to(device), lengths.to(device)).float()
        lg, tg, _ = restrict(groups, logits, target.to(device), group.to(device))
        tot += float(soft_ce(lg, tg, 0.0)) * len(x); n += len(x)
    return tot / max(1, n)


def scores(y, p, h, n):
    acc = lambda m: float((y[m] == p[m]).mean()) if m.any() else float("nan")
    f1 = []
    for c in range(n):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        if tp + fn: f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    hu, sy = acc(h == 1), acc(h == 0)
    return {"n": int(len(y)), "accuracy": acc(np.ones_like(y, bool)), "macro_f1": float(np.mean(f1)) if f1 else float("nan"),
            "human_accuracy": hu, "synthetic_accuracy": sy, "n_human": int((h == 1).sum()), "selection_score": float(np.nanmean([hu, sy]))}


def build_items(table, indices, keys, slots, humans, slotted, classes):
    cmds = table["command"].to_pylist(); items = []
    for i in indices:
        c = cmds[i]
        if c in slotted and f"{c}|{slots[i]}" in classes:
            items.append(("main", i, classes.index(f"{c}|{slots[i]}"), keys[i], humans[i], slotted.index(c)))
    return items


def final_eval(model, a, device, out, slotted, classes, groups):
    results, preds = {}, []
    noises = load_demand(heldout=True); envs = sorted(noises)
    for split in ("test", "holdout"):
        t = pq.read_table(sorted((a.ai231 / "data").glob(f"{split}-*.parquet")), columns=["file", "command", "slot_value", "is_synthetic", "audio"])
        keys = [f"{split}|{f}" for f in t["file"].to_pylist()]; humans = [int(s == 0) for s in t["is_synthetic"].to_pylist()]
        idx = [i for i, c in enumerate(t["command"].to_pylist()) if c in CLASSES]
        items = build_items(t, idx, keys, t["slot_value"].to_pylist(), humans, slotted, classes)[:64 if a.smoke else None]
        for cond in ("clean",) + NOISE_CONDS:
            noise = (noises, NOISE_SNRS[NOISE_CONDS.index(cond)]) if cond in NOISE_CONDS else None
            loader = DataLoader(SlotDataset(items, len(classes), {"main": t["audio"]}, fixed_noise=noise),
                                batch_size=128, num_workers=a.workers, collate_fn=collate)
            y, p, h = predict(model, loader, groups, device)
            results[f"{split}/{cond}"] = scores(y, p, h, len(classes))
            preds += [(split, cond, it[3], classes[yy], classes[pp], noise_env(it[3], envs) if noise else "") for it, yy, pp in zip(items, y, p)]
            print(split, cond, json.dumps(results[f"{split}/{cond}"]), flush=True)
    with (out / "predictions.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["split", "condition", "key", "target", "prediction", "noise_env"]); w.writerows(preds)
    results["test/noise_mean"] = {"accuracy": float(np.mean([results[f"test/{c}"]["accuracy"] for c in NOISE_CONDS]))}
    return results


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_ROOT / "ai231_filtered_lenient")
    p.add_argument("--ai231", type=Path, default=AI231)
    p.add_argument("--numerals", type=Path, default=DATA_ROOT / "numerals_subset.parquet")
    p.add_argument("--timings", type=Path, default=ARTIFACTS / "whisper/slot_word_timestamps.jsonl.gz")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--seed", type=int, default=23101)
    p.add_argument("--epochs", type=int, default=150); p.add_argument("--select-window", type=int, default=5)
    p.add_argument("--warmup-epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1); p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--gpu-fraction", type=float, default=0.5)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    a = parse(); out = a.output; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.smoke else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction, 0); torch.backends.cudnn.benchmark = True
    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - a.workers))
    slotted, classes, groups = slot_schema(a.ai231); n_out = len(classes)

    # speaker-disjoint split on the filtered train data
    main_files = sorted((a.data / "data").glob("train-*.parquet")); fil_files = sorted((a.data / "supplemental_fil").glob("train-*.parquet"))
    main, fil = read(main_files, True), read(fil_files, True)
    vs = set(check_validation_speakers(a.data, "slot_validation_speakers.json"))
    speakers = main["speaker_id"].to_pylist()
    mi = eligible(main)
    vi = [i for i in mi if speakers[i] in vs]
    ti = [i for i in mi if speakers[i] not in vs] + [len(main) + i for i in eligible(fil)]
    table = pa.concat_tables([main, fil])
    meta = pa.concat_tables([pq.read_table(f, columns=["file", "slot_value", "is_synthetic"]) for f in main_files + fil_files])
    keys = [f"{'train' if i < len(main) else 'supplemental_fil'}|{f}" for i, f in enumerate(meta["file"].to_pylist())]
    humans = [int(s == 0) for s in meta["is_synthetic"].to_pylist()]; slots = meta["slot_value"].to_pylist()
    train_items = build_items(table, ti, keys, slots, humans, slotted, classes)
    val_items = build_items(table, vi, keys, slots, humans, slotted, classes)
    sources = {"main": table["audio"]}
    num = pq.read_table(a.numerals, columns=["file", "slot_class", "audio"]); sources["num"] = num["audio"]
    for j, (f, sc) in enumerate(zip(num["file"].to_pylist(), num["slot_class"].to_pylist())):
        train_items.append(("num", j, classes.index(sc), f"numerals|{f}", 1, slotted.index(sc.split("|")[0])))
    if a.smoke:
        rng = np.random.default_rng(0)
        train_items = [train_items[i] for i in rng.choice(len(train_items), 256, replace=False)]
        val_items = [val_items[i] for i in rng.choice(len(val_items), 96, replace=False)]
        a.epochs, a.warmup_epochs, a.batch_size, a.workers = 2, 1, 32, 0
    timings = load_timings(a.timings)
    noises = load_noise_dir(SPEECH_NOISE)
    demand_train = load_demand(heldout=False)
    chunks = [c for cs in demand_train.values() for c in cs]  # equal weight: half Speech Commands, half DEMAND
    augmentor = WaveformAugmentor(noises * max(1, len(chunks) // len(noises)) + chunks)
    mk = lambda ds, shuffle: DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.workers, collate_fn=collate,
                                        pin_memory=device.type == "cuda", persistent_workers=a.workers > 0, drop_last=shuffle)
    tl = mk(SlotDataset(train_items, n_out, sources, True, augmentor, timings), True)
    vl = mk(SlotDataset(val_items, n_out, sources), False)

    model = SlotTCDSCNN(n_out, 0.30, 128).to(device)
    ema = AveragedModel(model, use_buffers=True, avg_fn=lambda av, cur, n: av * min(a.ema_decay, (1 + int(n)) / (10 + int(n))) + cur * (1 - min(a.ema_decay, (1 + int(n)) / (10 + int(n)))))
    decay = [p for p in model.parameters() if p.ndim > 1]; no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    total, warm = a.epochs * len(tl), a.warmup_epochs * len(tl)
    n_params = sum(p.numel() for p in model.parameters())
    config = {"model": "SlotTCDSCNN", "classes": classes, "seed": a.seed, "train_items": len(train_items), "val_items": len(val_items),
              "numerals_items": sum(it[0] == "num" for it in train_items), "timings": len(timings),
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "parameter_count": n_params,
              "selection": f"validation mean(human acc, synthetic acc); better of raw/EMA; centre of the best {a.select_window}-epoch moving average",
              "demand_train_envs": sorted(demand_train), "demand_eval_envs": list(DEMAND_HELDOUT), "steps_per_epoch": len(tl),
              "node": os.environ.get("SLURMD_NODENAME", os.uname().nodename), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "environment": collect_environment(device)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k: config[k] for k in ("train_items", "val_items", "numerals_items", "timings", "parameter_count")}), flush=True)

    step, history, t0 = 0, [], time.time()
    ep_dir = out / "epochs"; ep_dir.mkdir(exist_ok=True)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    for epoch in range(1, a.epochs + 1):
        model.train(); tot = n = 0
        for x, lengths, target, group, _ in tl:
            x, lengths, target, group = x.to(device), lengths.to(device), target.to(device), group.to(device)
            for g in opt.param_groups: g["lr"] = lr_at(step, total, warm, a.lr)
            x = spec_augment(x, lengths, time_masks=0)
            with torch.autocast(**amp):
                logits = model(x, lengths)
            lg, tg, _ = restrict(groups, logits.float(), target, group)
            loss = soft_ce(lg, tg, a.label_smoothing)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); ema.update_parameters(model); step += 1; tot += float(loss) * len(x); n += len(x)
        cand = {name: scores(*predict(net, vl, groups, device), n_out) for name, net in (("raw", model), ("ema", ema.module))}
        chosen = max(cand, key=lambda k: cand[k]["selection_score"]); v = cand[chosen]
        net = model if chosen == "raw" else ema.module
        row = {"epoch": epoch, "train_loss": tot / max(1, n), "val_loss": mean_loss(net, vl, groups, device), "chosen": chosen,
               **{"val_" + k: x for k, x in v.items()}, "seconds": time.time() - t0}
        history.append(row); print(json.dumps(row), flush=True)
        torch.save({"model_state": net.state_dict(), "weights": chosen, "epoch": epoch,
                    "model_config": {"num_classes": n_out, "dropout": 0.30, "width": 128}, "classes": classes,
                    "validation": v}, ep_dir / f"e{epoch:03d}.pt")
    W = min(a.select_window, len(history))
    smooth = np.convolve([r["val_selection_score"] for r in history], np.ones(W) / W, "valid")
    best_epoch = int(np.argmax(smooth)) + W // 2 + 1
    print(f"best epoch {best_epoch} (smoothed validation {smooth.max():.4f})", flush=True)
    shutil.copy(ep_dir / f"e{best_epoch:03d}.pt", out / "best.pt")
    with (out / "history.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=True); model.load_state_dict(ck["model_state"])
    res = final_eval(model, a, device, out, slotted, classes, groups)
    config.update(status="DONE", best_epoch=ck["epoch"], best_epoch_smoothed_validation=float(smooth.max()),
                  epochs_completed=len(history), optimizer_steps=step, training_seconds=time.time() - t0,
                  checkpoint_sha256=sha256_file(out / "best.pt"), validation=ck["validation"], final=res)
    (out / "config.json").write_text(json.dumps(config, indent=2)); (out / "COMPLETED").write_text("done\n")


if __name__ == "__main__":
    main()
