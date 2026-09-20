"""Tensor conversion at the embedding boundary; no model-loading dependencies."""
from typing import Any

import numpy as np
import torch


def embeddings_to_numpy(embeddings: Any) -> np.ndarray:
    """Detach tensors and convert to CPU float32 before crossing into NumPy."""
    if torch.is_tensor(embeddings):
        # NumPy cannot represent torch.bfloat16, and tensors requiring gradients
        # cannot be exported directly. Conversion must precede .numpy().
        return embeddings.detach().to(device="cpu", dtype=torch.float32).numpy()
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach()
        if torch.is_tensor(embeddings):
            return embeddings_to_numpy(embeddings)
        if getattr(embeddings, "is_cuda", False):
            embeddings = embeddings.cpu()
        return np.asarray(embeddings.numpy(), dtype=np.float32)
    if isinstance(embeddings, list):
        return np.vstack([
            embeddings_to_numpy(row) if hasattr(row, "detach") else np.asarray(row)
            for row in embeddings
        ]).astype(
            np.float32, copy=False
        )
    return np.asarray(embeddings, dtype=np.float32)
