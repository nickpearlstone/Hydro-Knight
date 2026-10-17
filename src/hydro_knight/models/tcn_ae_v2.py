"""
TCN autoencoder v2: dilated residual encoder, squeezed code, mirrored decoder.

Encoder: one residual block per dilation (dilations_for(window), e.g. 1, 2, 4, 8,
16, 13 for 90 steps) at constant width. The convolutions are unpadded ("valid"): the
input is the scored window plus `context` real steps on each side, and each block
trims 2 * dilation steps, so the encoder ends exactly on the scored window. Residual
connections add each block's input (center-cropped, or 1x1-projected when channels
change) to its output, which keeps deep stacks trainable.

Squeeze: three stride-2 convolutions shrink time (90 -> 45 -> 23 -> 12) while channels
halve toward the code size. Decoder: upsample + convolution back to 23, 45, 90 steps,
two padded residual blocks to restore detail, then a 1-wide convolution to the input
features. The anomaly score is the masked reconstruction error of the scored window.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..features.track_series import N_BASE, N_POSE
from ..features.windows_v2 import WindowSet, dilations_for

SIZES = {
    "small": (32, 8),
    "medium": (64, 16),
    "large": (128, 32),
}  # (width, code channels)


class DilatedBlock(nn.Module):
    """3-tap dilated convolution + ReLU with a residual connection.

    Unpadded by default (output shorter by 2 * dilation, residual center-cropped);
    pad=True keeps the length with reflect padding (used inside the decoder only).
    """

    def __init__(self, c_in: int, c_out: int, dilation: int, pad: bool = False):
        super().__init__()
        self.d, self.pad = dilation, pad
        self.conv = nn.Conv1d(
            c_in,
            c_out,
            3,
            dilation=dilation,
            padding=dilation if pad else 0,
            padding_mode="reflect",
        )
        self.skip = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)

    def forward(self, x):
        y = torch.relu(self.conv(x))
        s = self.skip(x)
        return y + (s if self.pad else s[..., self.d : -self.d])


def _halve(n: int) -> int:
    return (n + 1) // 2


class TCNAutoencoderV2(nn.Module):
    """Dilated residual TCN autoencoder over window + context; see module docstring."""

    def __init__(
        self,
        n_feat: int = N_BASE,
        width: int = 64,
        bottleneck: int = 16,
        window: int = 90,
    ):
        super().__init__()
        self.window = window
        self.dilations = dilations_for(window)
        self.context = sum(self.dilations)
        self.encoder = nn.Sequential(
            *[
                DilatedBlock(n_feat if i == 0 else width, width, d)
                for i, d in enumerate(self.dilations)
            ]
        )
        chans = [max(width // 2, bottleneck), max(width // 4, bottleneck), bottleneck]
        lens = [window]
        for _ in range(3):
            lens.append(_halve(lens[-1]))
        squeeze, c = [], width
        for i, co in enumerate(chans):
            squeeze.append(
                nn.Conv1d(c, co, 3, stride=2, padding=1, padding_mode="reflect")
            )
            if i < 2:
                squeeze.append(nn.ReLU())
            c = co
        self.squeeze = nn.Sequential(*squeeze)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for co, length in zip(
            [chans[1], chans[0], width], [lens[2], lens[1], lens[0]], strict=True
        ):
            self.ups.append(
                nn.Upsample(size=length, mode="linear", align_corners=False)
            )
            self.up_convs.append(nn.Conv1d(c, co, 3, padding=1, padding_mode="reflect"))
            c = co
        self.rebuild = nn.Sequential(
            DilatedBlock(width, width, 1, pad=True),
            DilatedBlock(width, width, 2, pad=True),
        )
        self.out = nn.Conv1d(width, n_feat, 1)

    def encode(self, x):
        """(B, F, window + 2*context) -> code (B, bottleneck, 12 for a 90-step window)."""
        return self.squeeze(self.encoder(x))

    def forward(self, x):
        """(B, F, window + 2*context) -> reconstruction of the scored window (B, F, window)."""
        z = self.encode(x)
        for up, conv in zip(self.ups, self.up_convs, strict=True):
            z = torch.relu(conv(up(z)))
        return self.out(self.rebuild(z))


def n_params(model: nn.Module) -> int:
    """Trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def masked_errors(pred, target, detected, coord_weight: float = 0.0):
    """Per-window weighted MSE; pose coordinates on missing steps weigh coord_weight.

    Args:
        pred, target: (B, F, T) tensors.
        detected: (B, T) bool, True where the swimmer was detected.
        coord_weight: weight of pose coordinates (features 0-33) on missing steps;
            their inputs are held placeholders with no ground truth.
    Returns:
        (B,) tensor.
    """
    w = torch.ones_like(target)
    d = detected.float()
    w[:, :N_POSE, :] = (d + (1.0 - d) * coord_weight)[:, None, :]
    return ((pred - target) ** 2 * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(
        1e-9
    )


def fit_scaler(ws: WindowSet, n: int = 2000, seed: int = 0):
    """Per-feature (mean, std) over the scored steps of up to n random windows."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(ws), size=min(n, len(ws)), replace=False)
    x, _ = ws.gather(rows)
    flat = x[:, ws.context : ws.context + ws.window].reshape(-1, x.shape[-1])
    std = flat.std(axis=0)
    # A (near-)constant feature gets scale 1: a tiny floor would blow rare values up.
    return flat.mean(axis=0), np.where(std < 1e-3, 1.0, std).astype(np.float32)


def _batch(ws: WindowSet, rows, scaler, device):
    x, det = ws.gather(rows)
    x = (x - scaler[0]) / scaler[1]
    xt = torch.tensor(np.transpose(x, (0, 2, 1)), dtype=torch.float32, device=device)
    return xt, torch.tensor(det, device=device)


def train_v2(
    ws: WindowSet,
    size: str = "medium",
    steps: int = 2000,
    batch: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    coord_weight: float = 0.0,
    device: str = "cpu",
):
    """Train on random mini-batches of normal windows; returns (model, scaler).

    The per-step loss is recorded on model.loss_history_.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    width, code = SIZES[size]
    model = TCNAutoencoderV2(ws.n_feat, width, code, ws.window).to(device)
    if model.context != ws.context:
        raise ValueError(f"model context {model.context} != window set {ws.context}")
    scaler = fit_scaler(ws, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    c, w = ws.context, ws.window
    history: list[float] = []
    model.train()
    for _ in tqdm(range(steps), desc=f"train {size}"):
        rows = rng.integers(0, len(ws), size=min(batch, len(ws)))
        x, det = _batch(ws, rows, scaler, device)
        loss = masked_errors(
            model(x), x[:, :, c : c + w], det[:, c : c + w], coord_weight
        ).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    model.loss_history_ = history
    return model, scaler


@torch.no_grad()
def score_v2(
    model,
    ws: WindowSet,
    scaler,
    coord_weight: float = 0.0,
    batch: int = 512,
    device="cpu",
):
    """Per-window masked reconstruction error (the anomaly score), shape (N,)."""
    model.eval()
    c, w = ws.context, ws.window
    out = []
    for a in range(0, len(ws), batch):
        x, det = _batch(ws, np.arange(a, min(a + batch, len(ws))), scaler, device)
        out.append(
            masked_errors(
                model(x), x[:, :, c : c + w], det[:, c : c + w], coord_weight
            ).cpu()
        )
    return torch.cat(out).numpy() if out else np.empty(0, np.float32)


def save_checkpoint(path, model: TCNAutoencoderV2, scaler, meta: dict | None = None):
    """Write weights, architecture, feature scaler and run metadata to one file.

    Args:
        path: output file (e.g. runs/<name>/model_small_s0.pt).
        model: a trained TCNAutoencoderV2.
        scaler: the (mean, std) arrays train_v2 returned.
        meta: anything needed to score with the model again (feature options, split).
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": {
                "n_feat": model.out.out_channels,
                "width": model.out.in_channels,
                "bottleneck": model.squeeze[-1].out_channels,
                "window": model.window,
            },
            "scaler": [np.asarray(scaler[0]), np.asarray(scaler[1])],
            "meta": meta or {},
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """Inverse of save_checkpoint: returns (model in eval mode, scaler, meta)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TCNAutoencoderV2(**ck["arch"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mean, std = ck["scaler"]
    return model, (mean.astype(np.float32), std.astype(np.float32)), ck["meta"]
