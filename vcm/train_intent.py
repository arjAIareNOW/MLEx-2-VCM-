"""Train the 19-intent Conformer on ai231_v2 and score it once on the official test and holdout splits.

Recipe (docs/04_intent_model.md):
  data      ai231_v2 train (+ supplemental_fil, supplemental_synth train voices, oversample.csv), speaker-disjoint 15 %
            validation (vcm.data); OUT_OF_SCOPE is not a target (handled by a reject threshold, vcm.reject_threshold)
  augment   vcm.augment.IntentAugmentor (waveform) + SpecAugment + mixup (alpha 0.3, p 0.5)
  optimiser AdamW (lr 2e-3, weight decay 0.05 on weights only), 5-epoch linear warm-up then per-step cosine to 1e-5,
            batch 128, 120 epochs max, gradient clip 5, bf16 autocast, label smoothing 0.1,
            class weights = inverse train frequency (mean 1)
  weights   EMA (decay min(0.999, (1+n)/(10+n))); each epoch the better of raw / EMA on validation is kept
  selection validation macro-F1 only; early stop after 30 epochs without improvement
  test      after training: test + holdout in-scope clips, never used for any choice

Usage: python -m vcm.train_intent --output runs/intent_conformer [--seed 23102] [--smoke]
"""
from __future__ import annotations

import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from .augment import IntentAugmentor, load_noise_dir, spec_augment
from .common import AI231, AI231_V2, CLASSES, SPEECH_NOISE, collect_environment, dump, set_reproducible, sha256_file
from .data import IntentDataset, official_split, prepare_intent, stats
from .features import collate
from .metrics import classification_report
from .models import IntentConformer, conformer_config


def lr_at(step: int, total: int, warmup: int, peak: float, floor: float = 1e-5) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return floor + (peak - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def make_loader(dataset, a, device, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.workers,
                      pin_memory=device.type == "cuda", collate_fn=collate, generator=torch.Generator().manual_seed(a.seed))


def class_weights(counts: dict[str, int], device) -> torch.Tensor:
    raw = torch.tensor([1.0 / counts[name] for name in CLASSES], dtype=torch.float32)
    return (raw / raw.mean()).to(device)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval(); targets, predictions, loss_sum, n = [], [], 0.0, 0
    for x, lengths, y in loader:
        logits = model(x.to(device, non_blocking=True), lengths.to(device, non_blocking=True))
        loss_sum += float(criterion(logits, y.to(device))) * y.numel(); n += y.numel()
        targets += y.tolist(); predictions += logits.argmax(1).cpu().tolist()
    return classification_report(targets, predictions, CLASSES), loss_sum / n


HISTORY_FIELDS = ("epoch", "learning_rate", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy",
                  "validation_macro_precision", "validation_macro_recall", "validation_macro_f1", "validation_weighted_f1")


def write_history(path: Path, history: list[dict]) -> None:
    import csv
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(history)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=AI231_V2, help="ai231_v2 folder")
    p.add_argument("--ai231", type=Path, default=AI231, help="original ai231 download (split check, test, holdout)")
    p.add_argument("--noise-dir", type=Path, default=SPEECH_NOISE)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=23102)
    p.add_argument("--epochs", type=int, default=120); p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.05); p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=0.3); p.add_argument("--mixup-prob", type=float, default=0.5)
    p.add_argument("--ema-decay", type=float, default=0.999); p.add_argument("--patience", type=int, default=30)
    p.add_argument("--width", type=int, default=96); p.add_argument("--dropout", type=float, default=0.30)
    p.add_argument("--trimpad-prob", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--gpu-fraction", type=float, default=0.5, help="cap on this process's share of GPU memory")
    p.add_argument("--smoke", action="store_true", help="2-epoch CPU check on 256 train / 128 validation / 64 test clips")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if a.smoke else "cuda")
    if not a.smoke:
        assert torch.cuda.is_available(), "CUDA required (use --smoke for a CPU check)"
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction, 0)

    table, ti, vi, detail = prepare_intent(a.data, a.ai231)
    set_reproducible(a.seed, a.cpu_threads)
    torch.backends.cudnn.deterministic = False; torch.backends.cudnn.benchmark = True  # speed over bitwise repro
    dump(out / "split_manifest.json", detail)
    if a.smoke:
        rng = np.random.default_rng(0)
        ti = sorted(rng.choice(ti, 256, replace=False).tolist()); vi = sorted(rng.choice(vi, 128, replace=False).tolist())
        a.epochs, a.warmup_epochs, a.batch_size, a.workers = 2, 1, 32, 0

    mapping = {c: i for i, c in enumerate(CLASSES)}
    aug = IntentAugmentor(load_noise_dir(a.noise_dir), a.trimpad_prob)
    tl = make_loader(IntentDataset(table, ti, mapping, aug), a, device, True)
    vl = make_loader(IntentDataset(table, vi, mapping), a, device, False)

    model = IntentConformer(19, a.dropout, a.width).to(device)

    def ema_avg(averaged, current, num_averaged):
        decay = min(a.ema_decay, (1 + int(num_averaged)) / (10 + int(num_averaged)))
        return averaged * decay + current * (1 - decay)
    ema = AveragedModel(model, avg_fn=ema_avg, use_buffers=True)
    weights = class_weights(detail["train"]["class_counts"], device)
    train_ce = nn.CrossEntropyLoss(weight=weights, label_smoothing=a.label_smoothing)
    eval_ce = nn.CrossEntropyLoss(weight=weights)
    decay = [p for p in model.parameters() if p.ndim > 1]; no_decay = [p for p in model.parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    total_steps = a.epochs * len(tl); warmup_steps = a.warmup_epochs * len(tl)
    params = sum(p.numel() for p in model.parameters())
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")

    config = {"status": "TRAINING", "model": "IntentConformer", "classes": CLASSES, "class_to_index": mapping,
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
              "architecture": conformer_config(a.width), "parameter_count": params, "float32_parameter_bytes": 4 * params,
              "split": {"train": len(ti), "validation": len(vi)},
              "augmentation": {"waveform": "trim/pad p.5; gain +/-6dB p.5; speed 0.9-1.1 p.5; shift +/-100ms p.5; noise 10-25dB p.5; extra noise 5-10dB p.3",
                               "spec_augment": "2 freq masks <=8 bins, 2 time masks <=10% len (<=40 frames)",
                               "mixup": {"alpha": a.mixup_alpha, "probability": a.mixup_prob}},
              "optimizer": {"name": "AdamW", "lr": a.lr, "weight_decay": a.weight_decay, "warmup_epochs": a.warmup_epochs,
                            "schedule": "linear warmup then per-step cosine to 1e-5", "label_smoothing": a.label_smoothing,
                            "ema_decay": a.ema_decay, "precision": "bf16 autocast" if device.type == "cuda" else "fp32",
                            "loss": "class-weighted cross-entropy (inverse train frequency, mean 1)"},
              "selection": "better of raw / EMA weights each epoch, validation macro-F1",
              "environment": collect_environment(device),
              "node": os.environ.get("SLURMD_NODENAME", os.uname().nodename), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    dump(out / "config.json", config)
    print(json.dumps({"parameters": params, "train": len(ti), "validation": len(vi), "steps_per_epoch": len(tl)}), flush=True)

    history, best, stale, step, start = [], -1.0, 0, 0, time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train(); loss_sum = correct = seen = 0
        for x, lengths, y in tl:
            x, lengths, y = x.to(device, non_blocking=True), lengths.to(device, non_blocking=True), y.to(device, non_blocking=True)
            for g in opt.param_groups: g["lr"] = lr_at(step, total_steps, warmup_steps, a.lr)
            x = spec_augment(x, lengths)
            lam = 1.0
            if a.mixup_alpha > 0 and float(torch.rand(())) < a.mixup_prob:
                lam = float(np.random.beta(a.mixup_alpha, a.mixup_alpha)); perm = torch.randperm(len(y), device=device)
                x = lam * x + (1 - lam) * x[perm]; lengths = torch.maximum(lengths, lengths[perm]); y2 = y[perm]
            with torch.autocast(**amp):
                logits = model(x, lengths)
            logits = logits.float()
            loss = train_ce(logits, y) if lam == 1.0 else lam * train_ce(logits, y) + (1 - lam) * train_ce(logits, y2)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); ema.update_parameters(model); step += 1
            loss_sum += float(loss) * len(y); correct += int((logits.argmax(1) == y).sum()); seen += len(y)
        candidates = {name: evaluate(net, vl, eval_ce, device) for name, net in (("raw", model), ("ema", ema.module))}
        chosen = max(candidates, key=lambda k: candidates[k][0].summary["macro_f1"])
        report, vloss = candidates[chosen]
        row = {"epoch": epoch, "learning_rate": opt.param_groups[0]["lr"], "train_loss": loss_sum / seen,
               "train_accuracy": correct / seen, "validation_loss": vloss, **{"validation_" + k: v for k, v in report.summary.items()}}
        history.append(row); write_history(out / "training_history.csv", history)
        print(json.dumps({**row, "selected_weights": chosen}), flush=True)
        if report.summary["macro_f1"] > best:
            best, stale = report.summary["macro_f1"], 0
            torch.save({"weights": chosen, "model_state": (model if chosen == "raw" else ema.module).state_dict(),
                        "model_config": {"arch": "conformer", "num_classes": 19, "dropout": a.dropout, "width": a.width},
                        "class_to_index": mapping, "epoch": epoch,
                        "validation_report": {"loss": vloss, **report.summary, "per_class": report.per_class,
                                              "confusion_matrix": report.confusion_matrix}}, out / "best.pt")
        else:
            stale += 1
            if stale >= a.patience:
                print(f"Early stop: no validation macro-F1 gain for {a.patience} epochs", flush=True); break

    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    config.update(status="TRAINED", best_epoch=ck["epoch"], epochs_completed=len(history), optimizer_steps=step,
                  training_wall_seconds=time.perf_counter() - start, validation_metrics=ck["validation_report"],
                  checkpoint_sha256=sha256_file(out / "best.pt"), checkpoint_size_bytes=(out / "best.pt").stat().st_size)
    if device.type == "cuda":
        config["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    dump(out / "config.json", config)

    # One-time scoring of the selected checkpoint on the official test split and the holdout (Pi benchmark set).
    best_model = IntentConformer(19, a.dropout, a.width).to(device); best_model.load_state_dict(ck["model_state"])
    results = {}
    for split in ("test", "holdout"):
        tab, indices = official_split(a.ai231, split)
        indices = indices[:64 if a.smoke else None]
        rep, rloss = evaluate(best_model, make_loader(IntentDataset(tab, indices, mapping), a, device, False), eval_ce, device)
        result = {"loss": rloss, **rep.summary, "per_class": rep.per_class, "confusion_matrix": rep.confusion_matrix,
                  "dataset": stats(tab, indices), "oos_excluded": len(tab) - len(indices)}
        dump(out / f"{split}_metrics.json", result)
        results[split] = {k: v for k, v in result.items() if k not in ("per_class", "confusion_matrix")}
        print(split, json.dumps(results[split]), flush=True)
    config.update(status="TRAINED_AND_EVALUATED", test_metrics=results["test"], holdout_metrics=results["holdout"])
    dump(out / "config.json", config)
    (out / "COMPLETED").write_text("done\n")


if __name__ == "__main__":
    main()
