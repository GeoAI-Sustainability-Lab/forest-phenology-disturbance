from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch import nn

CONTINUOUS_FEATURES = 18


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_scalers(
    arrays: dict[str, np.ndarray], indexes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(indexes) > 160_000:
        indexes = np.sort(np.random.default_rng(20260809).choice(indexes, 160_000, replace=False))
    subset = arrays["features"][indexes].astype(np.float32)
    uncensored = ~arrays["history_censor"][indexes].astype(bool)
    means = np.zeros(CONTINUOUS_FEATURES, dtype=np.float32)
    scales = np.ones(CONTINUOUS_FEATURES, dtype=np.float32)
    masks = (
        (subset[..., 18] > 0.5) & uncensored,
        (subset[..., 20] > 0) & uncensored,
        (subset[..., 19] > 0.5) & uncensored,
        (subset[..., 21] > 0) & uncensored,
    )
    for channel in range(CONTINUOUS_FEATURES):
        mask = masks[0 if channel < 6 else 1 if channel < 12 else 2 if channel < 15 else 3]
        values = subset[..., channel][mask]
        if values.size:
            means[channel] = float(values.mean())
            scales[channel] = max(float(values.std()), 1e-3)
    return means, scales


def prepare_features(
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    history: int,
    means: torch.Tensor,
    scales: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = arrays["features"][indexes, -history:].astype(np.float32)
    x = torch.from_numpy(values).to(device, non_blocking=True)
    support = x[..., -4:].clone()
    censored = torch.from_numpy(arrays["history_censor"][indexes, -history:].astype(bool)).to(
        device
    )
    support *= (~censored)[..., None].to(support.dtype)
    x[..., -4:] = support
    x[..., :CONTINUOUS_FEATURES] = (x[..., :CONTINUOUS_FEATURES] - means) / scales
    x[..., :6] *= support[..., 0:1]
    x[..., 6:12] *= (support[..., 2:3] > 0).to(x.dtype)
    x[..., 12:15] *= support[..., 1:2]
    x[..., 15:18] *= (support[..., 3:4] > 0).to(x.dtype)
    dates = torch.from_numpy(arrays["dates"][indexes, -history:].astype(np.int64)).to(device)
    target_days = torch.from_numpy(arrays["target_days"][indexes].astype(np.int64)).to(device)
    return x, dates, target_days


class SequenceResidual(nn.Module):
    def __init__(self, kind: str = "gru", hidden: int = 64) -> None:
        super().__init__()
        if kind not in {"gru", "transformer"}:
            raise ValueError(kind)
        self.kind = kind
        self.token = nn.Sequential(nn.Linear(26, hidden), nn.SiLU(), nn.LayerNorm(hidden))
        if kind == "gru":
            self.temporal = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, dropout=0.10)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 3,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.context = nn.Sequential(
            nn.Linear(hidden + 9, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.correction = nn.Linear(hidden, 6)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(
        self,
        x: torch.Tensor,
        dates: torch.Tensor,
        target_days: torch.Tensor,
        baseline: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed = (
            (x[..., -4] > 0.5) | (x[..., -3] > 0.5) | (x[..., -2] > 0) | (x[..., -1] > 0)
        ) & (dates >= 0)
        safe_dates = dates.clamp(min=0).float()
        phase = 2 * math.pi * torch.remainder(safe_dates, 365.2425) / 365.2425
        age = ((target_days[:, None].float() - safe_dates) / 365.2425).clamp(0, 5)
        tokens = self.token(
            torch.cat(
                [
                    x,
                    torch.sin(phase)[..., None],
                    torch.cos(phase)[..., None],
                    age[..., None],
                    observed.float()[..., None],
                ],
                dim=-1,
            )
        )
        if self.kind == "gru":
            encoded, _ = self.temporal(tokens)
            positions = (
                torch.where(
                    observed,
                    torch.arange(observed.shape[1], device=x.device)[None],
                    -1,
                )
                .max(dim=1)
                .values.clamp(min=0)
            )
            pooled = encoded[torch.arange(len(x), device=x.device), positions]
        else:
            encoded = self.temporal(tokens, src_key_padding_mask=~observed)
            pooled = (encoded * observed[..., None]).sum(dim=1) / observed.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
        target_phase = 2 * math.pi * torch.remainder(target_days.float(), 365.2425) / 365.2425
        last_day = torch.where(observed, safe_dates, torch.zeros_like(safe_dates)).max(dim=1).values
        gap = ((target_days.float() - last_day) / 365.2425).clamp(0, 2)
        context = torch.cat(
            [
                pooled,
                baseline,
                torch.sin(target_phase)[:, None],
                torch.cos(target_phase)[:, None],
                gap[:, None],
            ],
            dim=1,
        )
        correction = 0.08 * torch.tanh(self.correction(self.context(context)))
        return (baseline + correction).clamp(-0.05, 1.25), correction


def model_loss(
    prediction: torch.Tensor,
    correction: torch.Tensor,
    target: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    mae = torch.abs(prediction - target).mean()
    regret = torch.relu(
        torch.abs(prediction - target).mean(dim=1) - torch.abs(baseline - target).mean(dim=1)
    ).mean()
    dot = torch.sum(prediction * target, dim=1)
    denominator = torch.linalg.vector_norm(prediction, dim=1) * torch.linalg.vector_norm(
        target, dim=1
    )
    sam = torch.acos(torch.clamp(dot / denominator.clamp(min=1e-6), -1, 1)).mean()
    return mae + 0.025 * sam + 0.25 * regret + 0.01 * correction.abs().mean()


@torch.inference_mode()
def predict(
    model: SequenceResidual,
    arrays: dict[str, np.ndarray],
    indexes: np.ndarray,
    history: int,
    means: np.ndarray,
    scales: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    mean_t = torch.from_numpy(means).to(device)[None, None]
    scale_t = torch.from_numpy(scales).to(device)[None, None]
    for start in range(0, len(indexes), batch_size):
        selected = indexes[start : start + batch_size]
        x, dates, target_days = prepare_features(arrays, selected, history, mean_t, scale_t, device)
        baseline = torch.from_numpy(arrays["baseline_seasonal"][selected].astype(np.float32)).to(
            device
        )
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            prediction, _ = model(x, dates, target_days, baseline)
        outputs.append(prediction.float().cpu().numpy())
    return np.concatenate(outputs)
