# Issue 46: Remove redundant SentenceTransformersEmbedder and standardize on JinaV4

- Source: <https://github.com/r3d91ll/HADES-Lab/issues/46>
- Created: 2025-09-17
- Author: r3d91ll
- Status: open

## Problem Statement

The current `SentenceTransformersEmbedder` implementation is effectively hardcoded for the Jina v4 model:
- Lines 55-57: Uses Jina v4 token and dimension constants
- Line 82: Defaults to `jinaai/jina-embeddings-v4`
- Lines 94-95: Applies v4 settings even when the model differs

This mirrors the existing `JinaV4Embedder`, while adding additional maintenance overhead (~600 LOC).

## Architectural Context

The team is standardizing on a "one model, one RAG" approach, relying on Jina v4 embeddings. Maintaining a pseudo-generic wrapper introduces technical debt and confuses model selection.

## Proposed Solution

1. Archive `SentenceTransformersEmbedder` under the Acheron archive with a timestamped filename to retain history.
2. Update `core/embedders/embedders_factory.py`:
   - Default embedder type should be `jina`.
   - Drop sentence-transformers registration logic.
   - Add a factory docstring that explains the new standard workflow and how to extend with future embedders.
3. Update import sites that referenced `SentenceTransformersEmbedder` (for example tests and workflows) to use `JinaV4Embedder` directly or through the factory.

## Expected Benefits

- Removes redundant code and clarifies the single supported embedder.
- Simplifies configuration and reduces confusion about which embedder to use.
- Keeps the codebase aligned with the production architecture.

## Follow-up / Testing

- Confirm all workflows and tests operate correctly with `JinaV4Embedder`.
- Ensure factory detection logic returns the intended embedder after clean-up.
- Run benchmark comparisons if needed to ensure performance parity or improved throughput.
