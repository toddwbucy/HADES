"""CPU-only embedding export contracts; no model downloads or inference."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embedding.tensors import embeddings_to_numpy


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize("as_rows", [False, True])
def test_export_detaches_and_preserves_batch_values(dtype, as_rows):
    source = torch.tensor([[1.25, -2.5], [0.0, 4.0]], dtype=dtype, requires_grad=True)
    result = embeddings_to_numpy(list(source) if as_rows else source)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[1.25, -2.5], [0.0, 4.0]])
    assert source.requires_grad
    assert source.dtype == dtype
    assert source.grad is None


def test_mixed_tensor_and_array_rows_preserve_order():
    result = embeddings_to_numpy([
        torch.tensor([1.5, 2.0], dtype=torch.bfloat16),
        np.array([-3.0, 4.0], dtype=np.float64),
        [5.0, 6.0],
    ])
    np.testing.assert_array_equal(result, [[1.5, 2.0], [-3.0, 4.0], [5.0, 6.0]])
    assert result.dtype == np.float32


def test_float32_array_is_not_copied():
    source = np.array([[1.0, 2.0]], dtype=np.float32)
    assert embeddings_to_numpy(source) is source
