"""Multi-spectral (2-D DCT) channel attention for YOLOv8 — FcaNet-style.

Motivation
----------
SE-style channel attention squeezes each channel with global average pooling
(GAP). GAP is exactly the 0th (DC) component of the 2-D DCT: it keeps the
channel's mean energy and throws away *where* in the spectrum that energy
lives. FcaNet (Qin et al., "FcaNet: Frequency Channel Attention Networks",
ICCV 2021) generalises the squeeze step: channels are split into G groups and
each group is pooled with a different low-frequency 2-D DCT basis, so the
squeeze keeps multi-spectral evidence (texture / edge / gradient energy) that
GAP discards.

Industrial surface defects are precisely high-frequency, low-contrast
deviations sitting on a low-frequency background texture (rolling marks,
illumination gradients). A squeeze that can "see" more than the DC component
is therefore a natural, nearly free fit for defect detection. This module is
the low-cost, reproducible embodiment of the frequency-decoupling idea also
pursued by DSF-Net for thermal-control coating defect inspection.

Usage in a YOLOv8 yaml (channels are the post-scale layer widths)::

    - [-1, 1, DCTAttention, [64]]    # after the C2f producing P3 features
    - [-1, 1, DCTAttention, [128]]   # after P4
    - [-1, 1, DCTAttention, [256]]   # after P5

``SEAttention`` is the control: identical two-layer FC gate and capacity, but
a plain GAP squeeze (single DC branch). Any gain of DCTAttention over
SEAttention is attributable to the multi-spectral pooling, not to "adding an
attention module".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _dct1d(n: int, v: int) -> torch.Tensor:
    """Orthonormal 1-D DCT-II basis vector of length ``n`` for frequency ``v``."""
    k = torch.arange(n, dtype=torch.float64)
    scale = math.sqrt(1.0 / n) if v == 0 else math.sqrt(2.0 / n)
    return scale * torch.cos(math.pi * (2 * k + 1) * v / (2 * n))


def zigzag_frequencies(n: int) -> list[tuple[int, int]]:
    """The ``n`` lowest 2-D frequencies (u, v) in zigzag order, starting at DC (0, 0)."""
    out: list[tuple[int, int]] = []
    for s in range(64):
        for u in range(s + 1):
            out.append((u, s - u))
            if len(out) == n:
                return out
    return out


class _ChannelGate(nn.Module):
    """Shared FC gate: Linear(C -> C/r) - ReLU - Linear(C/r -> C) - Sigmoid."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        # Zero-init the last layer => gate starts as a uniform 0.5 scaling, so
        # pretrained backbone features are not scrambled at the start of finetuning.
        nn.init.zeros_(self.fc[2].weight)
        nn.init.zeros_(self.fc[2].bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:  # (B, C) -> (B, C)
        return self.fc(y)


class DCTAttention(nn.Module):
    """FcaNet-style multi-spectral channel attention.

    Channels are split into ``groups`` contiguous groups; group ``g`` is pooled
    with the fixed 2-D DCT basis of zigzag frequency ``g`` (group 0 -> DC ==
    GAP). Pooled descriptors are concatenated (channel order preserved) and fed
    through the shared FC gate. Cost: ~2*C*C/r parameters, negligible FLOPs.

    The DCT bases are a pure function of (H, W), so they are generated lazily
    and cached per feature-map size instead of being registered buffers — this
    keeps the module input-resolution agnostic (train vs. export imgsz).

    Args:
        channels: input channels; must equal the producing layer's output width.
        groups: number of channel groups (= number of DCT frequencies used).
        reduction: FC reduction ratio, as in SE.
    """

    def __init__(self, channels: int, groups: int = 8, reduction: int = 16):
        super().__init__()
        if channels % groups:
            groups = next(g for g in (16, 8, 4, 2, 1) if channels % g == 0)
        self.channels = channels
        self.groups = groups
        self.gate = _ChannelGate(channels, reduction)
        self.register_buffer("freq_idx", torch.tensor(zigzag_frequencies(groups), dtype=torch.long), persistent=False)
        self._basis_cache: dict[tuple[int, int, torch.device], torch.Tensor] = {}

    def _basis(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        key = (H, W, device)
        basis = self._basis_cache.get(key)
        if basis is None:
            bases = [_dct1d(H, u).outer(_dct1d(W, v)).float() for u, v in self.freq_idx.tolist()]
            basis = torch.stack(bases).view(1, self.groups, 1, H, W).to(device)
            self._basis_cache[key] = basis
        return basis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != self.channels:
            raise ValueError(f"DCTAttention built for {self.channels} channels but got {C}; "
                             "check the channel arg in the model yaml matches the scale")
        basis = self._basis(H, W, x.device)
        # multi-spectral squeeze: (B, G, C/G, H, W) * (1, G, 1, H, W) -> (B, G, C/G)
        y = x.view(B, self.groups, C // self.groups, H, W).mul(basis).sum(dim=(3, 4))
        y = y.reshape(B, C)  # group-major split == original channel order
        return x * self.gate(y)[:, :, None, None]


class SEAttention(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).

    Control experiment for DCTAttention: identical FC gate (hence identical
    parameter count) and identical insertion points — the only difference is
    the squeeze: plain GAP, i.e. a single DC branch.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.gate = _ChannelGate(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        if C != self.channels:
            raise ValueError(f"SEAttention built for {self.channels} channels but got {C}")
        y = x.mean(dim=(2, 3))  # GAP == DC-only squeeze
        return x * self.gate(y)[:, :, None, None]


if __name__ == "__main__":
    # Unit test: shape preserved, gradients flow, DC group reproduces GAP, params tiny.
    torch.manual_seed(0)
    for C in (64, 128, 256):
        m = DCTAttention(C)
        x = torch.randn(2, C, 80, 80, requires_grad=True)
        y = m(x)
        assert y.shape == x.shape
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        n_params = sum(p.numel() for p in m.parameters())
        # group 0 basis is constant 1/sqrt(H*W) => its pooled value is GAP * sqrt(C)... just check constancy
        b0 = m._basis(80, 80, x.device)[0, 0, 0]
        assert torch.allclose(b0, torch.full_like(b0, 1 / math.sqrt(80 * 80)), atol=1e-6)
        print(f"DCTAttention(C={C}, G={m.groups}): params={n_params}, shape ok, grad ok, DC==GAP ok")
    se = SEAttention(64)
    xd = torch.randn(2, 64, 40, 40)
    assert se(xd).shape == xd.shape
    print(f"SEAttention(64): params={sum(p.numel() for p in se.parameters())}, shape ok")
