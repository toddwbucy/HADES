#!/usr/bin/env python3
"""Verify a local Jina v4 snapshot, including every task-adapter tensor.

Run with the services Python environment. This loads the full model and performs
one short embedding; it does not establish a safe maximum document size.
"""
import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
from safetensors import safe_open
import torch

from embedding.jina_v4 import JinaV4Embedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = args.model.resolve(strict=True)
    checkpoint_path = root / 'adapters/adapter_model.safetensors'
    if not checkpoint_path.is_file():
        parser.error('model must be a local snapshot containing adapter weights')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; refusing to validate a CPU fallback')

    warnings.filterwarnings('error', message='Found missing adapter keys.*')
    started = time.monotonic()
    backend = JinaV4Embedder(model_name=str(root), device=args.device,
                            use_fp16=args.device.startswith('cuda'), batch_size=1)
    try:
        vectors = backend.embed_texts(['HADES installation compatibility check.'])
        if vectors.shape != (1, 2048) or not np.isfinite(vectors).all():
            raise RuntimeError('Expected one finite 2048-dimensional embedding')
        state = backend.model.state_dict()
        checked = 0
        with safe_open(checkpoint_path, framework='pt') as checkpoint:
            for key in checkpoint.keys():
                candidates = [key, key.replace('.lora_A.', '.lora_A.default.')
                              .replace('.lora_B.', '.lora_B.default.')]
                actual = next((state[k] for k in candidates if k in state), None)
                if actual is None:
                    raise RuntimeError(f'Missing adapter tensor: {key}')
                expected = checkpoint.get_tensor(key).to(actual.dtype)
                if not torch.equal(actual.cpu(), expected):
                    raise RuntimeError(f'Adapter values differ: {key}')
                checked += 1
        if not checked:
            raise RuntimeError('Adapter checkpoint contains no tensors')
        print(json.dumps({'device': backend.device, 'shape': list(vectors.shape),
                          'finite': True, 'adapter_tensors_verified': checked,
                          'seconds': time.monotonic() - started}))
    finally:
        backend.unload()


if __name__ == '__main__':
    main()
