import math
import struct

import numpy as np
import pytest
import torch

from tools.artifact.codecs.nvfp4 import decode_nvfp4_words
from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert.model import Model, Parameter
from tools.convert.pipeline import convert
from tools.convert.quantization.nvfp4 import quantize_blocks
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source


def _scale(code):
    exponent, mantissa = code >> 3, code & 7
    return math.ldexp(1 + mantissa / 8, exponent - 7) if exponent else mantissa / 512


def test_e2m1_midpoints_signed_zero_and_zero_groups():
    source = torch.tensor([[6, .25, .75, 1.25, 1.75, 2.5, 3.5, 5,
                            -6, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -0.0],
                           [0.] * 16])
    result = quantize_blocks(source, 1)
    assert result.scales[:, 0].tolist() == [0x38, 0]
    expected = [7, 0, 2, 2, 4, 4, 6, 6, 15, 8, 10, 10, 12, 12, 14, 8]
    packed = result.codes[0].tolist()
    assert [n for b in packed for n in (b & 15, b >> 4)] == expected
    assert result.codes[1].tolist() == [0] * 8


def test_quantized_words_against_independent_scalar_oracle():
    rng = np.random.default_rng(1234)
    source = rng.normal(size=(128, 64)).astype(np.float32)
    divisor = np.float32(2688 / np.abs(source).max())
    result = quantize_blocks(torch.from_numpy(source), float(divisor))
    scales = [_scale(c) for c in range(127)]
    levels = [0, .5, 1, 1.5, 2, 3, 4, 6]
    for row in (0, 1, 63, 127):
        for group in range(4):
            values = source[row, group * 16:(group + 1) * 16]
            raw = float(np.float32(float(np.abs(values).max()) * float(divisor) / 6))
            scale_code = min(range(127), key=lambda c: (abs(scales[c] - raw), c & 1))
            assert result.scales[row, group].item() == scale_code
            for j, value in enumerate(values):
                # The declared normalization boundary is binary32.
                normalized = float(np.float32(float(value) * float(divisor) / scales[scale_code]))
                code = min(range(8), key=lambda c: (abs(levels[c] - abs(normalized)), c & 1))
                if np.signbit(value):
                    code |= 8
                offset = group * 16 + j
                packed = result.codes[row, offset // 2].item()
                assert ((packed >> (4 * (offset % 2))) & 15) == code


def test_streamed_parent_keeps_one_divisor_and_use_auxiliary(tmp_path):
    values = torch.cat([torch.full((128, 64), 6.), torch.full((128, 64), 2688.)])
    model = Model({'text': {'config': {}}})
    model.add(Parameter('left', (128, 64), array_source(values[:128], 'left'), inputs=('input',)))
    model.add(Parameter('right', (128, 64), array_source(values[128:], 'right'), inputs=('input',)))
    model.packing_groups = [('left', 'right')]
    recipe = Recipe(model)
    recipe.assign(('left', 'right'), format='nvfp4', method='nvfp4_block_maxabs', activation_policy='AllowA4')
    path = tmp_path / 'weights.ninfer'
    convert(model, recipe, path, device='cpu', rows_per_chunk=128)
    with Artifact(path) as artifact:
        object_id = binding_parts(artifact.directory.bindings['left'], artifact.by_id)[0][0]
        assert binding_parts(artifact.directory.bindings['right'], artifact.by_id)[0][0] == object_id
        codes, scales, divisor = decode_nvfp4_words(artifact.read_object(object_id), (256, 64))
        assert divisor.item() == 1
        assert bool((codes == 0x77).all())
        assert bool((scales[:128] == 0x38).all())
        assert bool((scales[128:] == 0x7e).all())
        use = artifact.directory.uses[0]
        auxiliary = use['auxiliaries']['activation_input_divisor']
        assert artifact.read_object(auxiliary['object']) == struct.pack('<f', 1.)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_source_rejected(value):
    with pytest.raises(ValueError):
        quantize_blocks(torch.full((1, 16), value), 1)
