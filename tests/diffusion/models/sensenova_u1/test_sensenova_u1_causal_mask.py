# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the SenseNova-U1 block causal mask construction.

``create_block_causal_mask`` is the shared mask producer for the model's
attention. On NPU it must return a bool mask with True=attend — the dtype the
platform-default FLASH_ATTN (mindiesd) backend consumes; an additive float
mask passed through unconverted silently corrupts the output. Non-NPU
platforms keep the historical additive 0.0/-inf float mask. The inline mask
built in ``SenseNovaU1Model.forward`` mirrors the same contract.
"""

import pytest
import torch

from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import create_block_causal_mask

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _attend_set(mask: torch.Tensor) -> torch.Tensor:
    """Normalize either mask convention to a bool True=attend tensor."""
    return mask == 0.0 if mask.is_floating_point() else mask


def test_block_causal_mask_attend_set_is_lower_triangular():
    """Same semantics on every platform: a token attends to itself, earlier
    tokens sharing the same time index, and nothing later."""
    index = torch.arange(8)
    attend = _attend_set(create_block_causal_mask(index))
    assert attend.shape == (1, 1, 8, 8)
    assert torch.equal(attend[0, 0], torch.ones(8, 8, dtype=torch.bool).tril())


def test_block_causal_mask_groups_by_time_index():
    """Only tokens with the same index can attend to each other."""
    index = torch.tensor([0, 0, 1, 1])
    attend = _attend_set(create_block_causal_mask(index))[0, 0]
    assert not attend[2, 0].item()  # later group, earlier index
    assert attend[3, 1].item()
    assert not attend[0, 2].item()


def test_block_causal_mask_platform_dtype_contract():
    """NPU: bool True=attend (the mindiesd contract). Other platforms keep the
    additive float mask the SDPA fallback consumes."""
    from vllm_omni.platforms import current_omni_platform

    mask = create_block_causal_mask(torch.arange(4))
    if current_omni_platform.is_npu():
        assert mask.dtype == torch.bool
        assert mask[0, 0, 0, 0].item()  # True=attend, not True=masked
    else:
        assert mask.dtype == torch.float32
        assert mask[0, 0, 0, 0].item() == 0.0  # additive 0.0/-inf form
        assert mask.min().item() == float("-inf")
