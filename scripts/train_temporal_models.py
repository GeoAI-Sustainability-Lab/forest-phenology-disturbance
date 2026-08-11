from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from forest_phenology_disturbance.data import load_feature_shards
from forest_phenology_disturbance.models import (
    SequenceResidual,
    compute_scalers,
    model_loss,
    predict,
    prepare_features,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUTPUT = ROOT / "outputs" / "metrics"


def sample_metrics(truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mae = np.abs(prediction - truth).mean(axis=1)
    dot = np.sum(truth * prediction, axis=1)
    denominator = np.linalg.norm(truth, axis=1) * np.linalg.norm(prediction, axis=1)
    sam = np.degrees(np.arccos(np.clip(dot / np.maximum(denominator, 1e-8), -1.0, 1.0)))
    return mae, sam


def group_score(
    arrays: dict[str, np.ndarray], indexes: np.ndarray, prediction: np.ndarray
) -> float:
    truth = arrays["targets"][indexes].astype(np.float32)
    mae, sam = sample_metrics(truth, prediction)
    groups = arrays["sample_group_ids"][indexes]
    h1 = arrays["horizon_seasons"][indexes] == 1
    group_ids = np.unique(groups[h1])
    group_mae = np.mean([mae[(groups == group) & h1].mean() for group in group_ids])
    group_sam = np.mean([sam[(groups == group) & h1].mean() for group in group_ids])
    return float(group_mae + 0.00025 * group_sam)


def choose_alpha(
    arrays: dict[str, np.ndarray], indexes: np.ndarray, raw: np.ndarray
) -> tuple[float, float]:
    baseline = arrays["baseline_seasonal"][indexes].astype(np.float32)
    candidates = []
    for alpha in np.linspace(0, 1, 21):
        prediction = baseline + float(alpha) * (raw - baseline)
        candidates.append((group_score(arrays, indexes, prediction), float(alpha)))
    score, alpha = min(candidates)
    return alpha, score


def train_seed(
    arrays: dict[str, np.ndarray],
    model_kind: str,
    history: int,
    fold: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    set_seed(seed)
    train = np.where(
        (arrays["folds"] != fold)
        & (arrays["target_years"] >= 2019)
        & (arrays["target_years"] <= 2022)
    )[0]
    validation = np.where((arrays["folds"] != fold) & (arrays["target_years"] == 2023))[0]
    test = np.where((arrays["folds"] == fold) & np.isin(arrays["target_years"], [2024, 2025]))[0]
    if args.smoke:
        rng = np.random.default_rng(seed)
        train = np.sort(rng.choice(train, min(24_000, len(train)), replace=False))
        validation = np.sort(rng.choice(validation, min(6_000, len(validation)), replace=False))
        test = np.sort(rng.choice(test, min(6_000, len(test)), replace=False))

    means, scales = compute_scalers(arrays, train)
    model = SequenceResidual(kind=model_kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    mean_t = torch.from_numpy(means).to(device)[None, None]
    scale_t = torch.from_numpy(scales).to(device)[None, None]
    best_score = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience = 0
    started = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train)
        for start in range(0, len(order), args.batch_size):
            selected = order[start : start + args.batch_size]
            x, dates, target_days = prepare_features(
                arrays, selected, history, mean_t, scale_t, device
            )
            target = torch.from_numpy(arrays["targets"][selected].astype(np.float32)).to(device)
            baseline = torch.from_numpy(
                arrays["baseline_seasonal"][selected].astype(np.float32)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                prediction, correction = model(x, dates, target_days, baseline)
                loss = model_loss(prediction, correction, target, baseline)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        raw_validation = predict(
            model, arrays, validation, history, means, scales, args.eval_batch_size, device
        )
        validation_score = group_score(arrays, validation, raw_validation)
        print(
            f"fold={fold} seed={seed} epoch={epoch + 1} validation={validation_score:.8f}",
            flush=True,
        )
        if validation_score < best_score - 2e-6:
            best_score = validation_score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    raw_validation = predict(
        model, arrays, validation, history, means, scales, args.eval_batch_size, device
    )
    raw_test = predict(model, arrays, test, history, means, scales, args.eval_batch_size, device)
    MODELS.mkdir(parents=True, exist_ok=True)
    checkpoint = MODELS / f"{model_kind}_h{history}_fold{fold}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "means": means,
            "scales": scales,
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        checkpoint,
    )
    return (
        raw_validation,
        raw_test,
        {
            "fold": fold,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_score": best_score,
            "minutes": (time.time() - started) / 60.0,
            "validation_indexes": validation,
            "test_indexes": test,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gru", "transformer"], default="gru")
    parser.add_argument("--history", type=int, choices=[8, 16], default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4707, 4721, 4733])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    arrays = load_feature_shards()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} samples={len(arrays['targets'])} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"
    )
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    audit_rows: list[dict[str, object]] = []
    for fold in args.folds:
        validation_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        validation_indexes: np.ndarray | None = None
        test_indexes: np.ndarray | None = None
        for seed in args.seeds[:1] if args.smoke else args.seeds:
            raw_validation, raw_test, audit = train_seed(
                arrays, args.model, args.history, fold, seed, args, device
            )
            validation_parts.append(raw_validation)
            test_parts.append(raw_test)
            validation_indexes = audit.pop("validation_indexes")
            test_indexes = audit.pop("test_indexes")
            audit_rows.append(audit)
        assert validation_indexes is not None and test_indexes is not None
        mean_validation = np.mean(validation_parts, axis=0)
        mean_test = np.mean(test_parts, axis=0)
        alpha, score = choose_alpha(arrays, validation_indexes, mean_validation)
        baseline = arrays["baseline_seasonal"][test_indexes].astype(np.float32)
        prediction = baseline + alpha * (mean_test - baseline)
        parts.append((test_indexes, prediction.astype(np.float32)))
        print(f"fold={fold} selected_alpha={alpha:.2f} validation={score:.8f}")

    indexes = np.concatenate([part[0] for part in parts])
    predictions = np.concatenate([part[1] for part in parts])
    order = np.argsort(indexes)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / f"refit_{args.model}_h{args.history}_oof.npz",
        global_index=indexes[order],
        truth=arrays["targets"][indexes[order]].astype(np.float32),
        prediction=predictions[order],
    )
    summary = {
        "model": args.model,
        "history": args.history,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "smoke": args.smoke,
        "folds": args.folds,
        "seeds": args.seeds[:1] if args.smoke else args.seeds,
        "runs": audit_rows,
    }
    (OUTPUT / f"refit_{args.model}_h{args.history}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
