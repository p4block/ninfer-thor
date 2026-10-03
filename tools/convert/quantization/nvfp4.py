"""Offline E2M1 weight quantization with K16 E4M3FN block scales."""

from dataclasses import dataclass
import struct

import numpy as np
import torch

from .fp8_row import _round_e4m3fn_rne

_VALUES = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Nvfp4Words:
    codes: torch.Tensor
    scales: torch.Tensor
    weight_divisor: bytes


def quantize_blocks(weight: torch.Tensor, divisor: float) -> Nvfp4Words:
    """Quantize floating [N,K]; caller supplies one FP32 divisor for its parent.

    Represented weights are E2M1 * E4M3FN / divisor. Both rounding steps use
    nearest, ties to even; block scales round before weight-code selection.
    """
    if weight.dim() != 2 or not weight.dtype.is_floating_point:
        raise TypeError('NVFP4 source must be a floating-point matrix')
    if weight.shape[0] <= 0 or weight.shape[1] <= 0 or weight.shape[1] % 16:
        raise ValueError('NVFP4 source requires positive dimensions and K%16=0')
    divisor = np.float32(divisor)
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError('NVFP4 divisor must be finite positive FP32')
    host = weight.detach().to(device='cpu', dtype=torch.float32).numpy()
    if not np.isfinite(host).all():
        raise ValueError('NVFP4 source contains NaN or infinity')
    groups = host.reshape(host.shape[0], -1, 16)
    amax = np.max(np.abs(groups), axis=-1)
    raw = (amax.astype(np.float64) * float(divisor) / 6).astype(np.float32)
    scales = _round_e4m3fn_rne(np.clip(raw, 0, 448))
    # Preserve tiny nonzero groups rather than silently rounding their scale to zero.
    scales[(amax != 0) & (scales == 0)] = 1
    scale_values = torch.from_numpy(scales).view(torch.float8_e4m3fn).float().numpy()
    normalized = np.zeros_like(groups)
    np.divide(groups.astype(np.float64) * float(divisor),
              scale_values[..., None], out=normalized, where=scale_values[..., None] != 0)
    magnitude = np.clip(np.abs(normalized), 0, 6)
    upper = np.minimum(np.searchsorted(_VALUES, magnitude, side='left'), 7)
    lower = np.maximum(upper - 1, 0)
    lo = magnitude - _VALUES[lower]
    hi = _VALUES[upper] - magnitude
    choose_upper = (hi < lo) | ((hi == lo) & ((upper & 1) == 0))
    codes = np.where(choose_upper, upper, lower).astype(np.uint8)
    codes |= np.where(np.signbit(groups), 8, 0).astype(np.uint8)
    codes = codes.reshape(host.shape)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return Nvfp4Words(torch.from_numpy(packed), torch.from_numpy(scales),
                      struct.pack('<f', float(divisor)))
