#!/usr/bin/env python3
"""
Jina v4 Embedder with proper API usage, fp16 support, and LATE CHUNKING.

Late chunking preserves contextual relationships across text boundaries.
The 32k token context window allows processing entire documents at once,
then intelligently chunking while preserving cross-boundary semantic relationships.
"""

# cspell:ignore jina Jina embedder Embedder

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .embedders_base import EmbedderBase, EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass
class ChunkWithEmbedding:
    """
    Represents a text chunk with its late-chunked embedding.

    The embedding preserves awareness of surrounding context from the
    full document, enabling superior semantic search compared to
    independently embedded chunks.
    """
    text: str
    embedding: np.ndarray
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    chunk_index: int
    total_chunks: int
    context_window_used: int  # How many tokens were in the full context


class JinaV4Embedder(EmbedderBase):
    """
    Jina v4 embedder with late chunking support.
    
    This embedder supports both:
    1. Traditional embedding (for short texts, backward compatibility)
    2. Late chunking (for long documents, superior context preservation)
    """

    # Jina v4 constants
    MAX_TOKENS = 32768  # Jina v4's context window
    EMBEDDING_DIM = 2048

    def __init__(self, config: EmbeddingConfig | dict[str, Any] | None = None, **kwargs: Any) -> None:
        """
        Initialize Jina v4 embedder with late chunking support.

        Args:
            config: EmbeddingConfig object or dict/None for defaults
            **kwargs: Additional configuration overrides
        """
        # Build config dict from various input formats
        config_dict: dict[str, Any] = {}

        if config is None:
            config_dict = {}
        elif isinstance(config, EmbeddingConfig):
            # Extract values from EmbeddingConfig object
            config_dict = {
                'device': config.device,
                'use_fp16': config.use_fp16,
                'batch_size': config.batch_size,
                'chunk_size_tokens': config.chunk_size_tokens,
                'chunk_overlap_tokens': config.chunk_overlap_tokens,
                'model_name': config.model_name,
                'max_seq_length': config.max_seq_length,
            }
        elif isinstance(config, dict):
            config_dict = config.copy()
        else:
            # Old-style single param (device) - for backwards compatibility
            config_dict = {'device': str(config)}  # type: ignore[unreachable]

        # Apply kwargs overrides
        config_dict.update(kwargs)

        # Determine device with hard CPU fallback
        requested_device = config_dict.get('device', 'cuda')
        if requested_device.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, forcing CPU fallback")
            self.device = 'cpu'
        else:
            self.device = requested_device

        self.model_name = config_dict.get('model_name', 'jinaai/jina-embeddings-v4')
        self.batch_size = config_dict.get('batch_size', 128)  # Default to 128 for better throughput

        # Create proper EmbeddingConfig for base class
        base_config = EmbeddingConfig(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            max_seq_length=config_dict.get('max_seq_length', self.MAX_TOKENS),
            use_fp16=config_dict.get('use_fp16', True),
            chunk_size_tokens=config_dict.get('chunk_size_tokens', 500),
            chunk_overlap_tokens=config_dict.get('chunk_overlap_tokens', 200),
        )

        # Initialize base class
        super().__init__(base_config)

        # Set chunking parameters from config
        self.chunk_size_tokens = config_dict.get('chunk_size_tokens', 500)
        self.chunk_overlap_tokens = config_dict.get('chunk_overlap_tokens', 200)
        use_fp16 = config_dict.get('use_fp16', True)

        # Load model with appropriate dtype
        # Check if device starts with "cuda" to handle cuda:0, cuda:1, etc.
        dtype = torch.float16 if (use_fp16 and self.device.startswith("cuda")) else torch.float32

        logger.info(f"Loading {self.model_name} on {self.device} with dtype={dtype}")
        logger.info(f"Batch size for embedding: {self.batch_size}")
        logger.info(f"Late chunking config: {self.chunk_size_tokens} tokens/chunk, {self.chunk_overlap_tokens} overlap")

        # Load tokenizer first for late chunking
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model and then move to target device
        # device_map should be "auto" or a dict, not a device string
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=dtype
        )

        # Move model to the target device if not CPU
        if self.device and self.device != "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()

        # Log actual model configuration
        logger.info("Jina v4 model loaded with late chunking support")
        logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
        logger.info("Embedding dimension: %s", self.EMBEDDING_DIM)
        logger.info(f"Model has encode method: {hasattr(self.model, 'encode')}")

        # Check if Flash Attention is available and being used
        try:
            import flash_attn
            logger.info("Flash Attention 2 is available - model should use it automatically")
        except ImportError:
            logger.warning("Flash Attention 2 not available - performance may be limited")

    def embed_texts(self,
                    texts: list[str],
                    task: str = "retrieval.passage",
                    batch_size: int | None = None) -> np.ndarray:
        """
        Embed texts using Jina v4.
        
        Args:
            texts: List of texts to embed
            task: Task type (retrieval, text-matching, code)
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (N x 2048)
        """
        all_embeddings = []
        batch_size = batch_size or self.batch_size

        # Commented out for performance - this was logging 30+ times per second
        # logger.info(f"Processing {len(texts)} texts with batch_size={batch_size}")

        with torch.no_grad():
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]

                # Use Jina's encode method if available
                if hasattr(self.model, 'encode'):
                    # Jina v4 encode method accepts task in {'retrieval','text-matching','code'}
                    # and prompt_name in {'query','passage'} for retrieval prefix.
                    jina_task_map = {
                        'retrieval.passage': 'retrieval',
                        'retrieval.query': 'retrieval',
                        'retrieval': 'retrieval',
                        'text-matching': 'text-matching',
                        'code': 'code',
                    }
                    jina_task = jina_task_map.get(task, 'retrieval')
                    prompt_name = 'query' if task == 'retrieval.query' else 'passage'
                    embeddings = self.model.encode_text(
                        batch,
                        task=jina_task,
                        prompt_name=prompt_name,
                    )
                else:
                    # Jina v4 requires task_label when using forward pass.
                    # Valid LoRA adapters are: 'retrieval', 'text-matching', 'code'
                    # (query vs passage is a prompt prefix, not a separate adapter)
                    task_mapping = {
                        'retrieval.passage': 'retrieval',
                        'retrieval.query': 'retrieval',
                        'retrieval': 'retrieval',
                        'text-matching': 'text-matching',
                        'code': 'code',
                    }
                    task_label = task_mapping.get(task, 'retrieval')

                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.MAX_TOKENS
                    ).to(self.device)

                    # Add task_label to inputs - must be a string for LoRA adapter selection
                    with torch.no_grad():
                        outputs = self.model(**inputs, task_label=task_label)

                        # Jina v4 returns single_vec_emb for 2048-dimensional embeddings
                        if hasattr(outputs, 'single_vec_emb') and outputs.single_vec_emb is not None:
                            embeddings = outputs.single_vec_emb
                        else:
                            # Log error with available attributes for debugging
                            available_attrs = [attr for attr in dir(outputs) if not attr.startswith('_')]
                            raise AttributeError(
                                f"Expected 'single_vec_emb' in JinaV4 output, but got: {available_attrs}. "
                                f"Output type: {type(outputs).__name__}"
                            )

                        if torch.is_tensor(embeddings):
                            if embeddings.is_cuda:
                                embeddings = embeddings.cpu()
                            embeddings = embeddings.numpy()

                # Debug: Check what type of object we got
                # logger.debug(f"Embeddings type: {type(embeddings)}")

                # Handle different return types - prioritize torch.is_tensor check
                if torch.is_tensor(embeddings):
                    # Handle PyTorch tensors directly
                    if embeddings.is_cuda:
                        embeddings = embeddings.cpu()
                    embeddings = embeddings.numpy()
                elif hasattr(embeddings, 'detach'):  # Other tensor-like objects
                    embeddings = embeddings.detach()
                    if hasattr(embeddings, 'is_cuda') and embeddings.is_cuda:
                        embeddings = embeddings.cpu()
                    embeddings = embeddings.numpy()
                elif isinstance(embeddings, list):
                    # If it's a list of tensors
                    processed = []
                    for e in embeddings:
                        if torch.is_tensor(e):
                            if e.is_cuda:
                                e = e.cpu()
                            processed.append(e.numpy())
                        elif hasattr(e, 'detach'):
                            e = e.detach()
                            if hasattr(e, 'is_cuda') and e.is_cuda:
                                e = e.cpu()
                            processed.append(e.numpy())
                        else:
                            processed.append(np.array(e))
                    embeddings = np.vstack(processed)
                elif not isinstance(embeddings, np.ndarray):
                    # Try to convert to numpy
                    try:
                        embeddings = np.array(embeddings)
                    except Exception as e:
                        logger.error(f"Cannot convert embeddings of type {type(embeddings)} to numpy: {e}")
                        raise

                all_embeddings.append(embeddings.astype(np.float32, copy=False))

                # DO NOT clear GPU cache after each batch - this kills performance!
                # PyTorch's allocator efficiently reuses memory.
                # Only clear cache if encountering OOM errors.
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()

        # Concatenate all batches
        if all_embeddings:
            result = np.vstack(all_embeddings).astype(np.float32, copy=False)
        else:
            result = np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        return result

    def embed_single(self, text: str, task: str = "retrieval.passage") -> np.ndarray:
        """
        Embed a single text (required by EmbedderBase interface).

        Args:
            text: Text to embed
            task: Task type

        Returns:
            1D embedding array
        """
        embeddings = self.embed_texts([text], task=task, batch_size=1)
        return embeddings[0] if embeddings.size > 0 else np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.EMBEDDING_DIM

    @property
    def max_sequence_length(self) -> int:
        """Get the maximum sequence length supported."""
        return self.MAX_TOKENS

    @property
    def supports_late_chunking(self) -> bool:
        """Whether this embedder supports late chunking."""
        return True

    @property
    def supports_multimodal(self) -> bool:
        """Whether this embedder supports multimodal inputs."""
        return True  # Jina v4 supports images

    def embed_code(self,
                   code_snippets: list[str],
                   batch_size: int = 4) -> np.ndarray:
        """
        Embed code using the code-specific task.
        
        Args:
            code_snippets: List of code snippets
            batch_size: Batch size for processing
            
        Returns:
            Numpy array of embeddings (N x 2048)
        """
        return self.embed_texts(code_snippets, task="code", batch_size=batch_size)

    def embed_images(self, images: list[bytes | Image.Image | str]) -> np.ndarray:
        """
        Embed images using Jina v4's multimodal capabilities.
        
        Args:
            images: List of images as bytes, PIL Images, or base64 strings
            
        Returns:
            L2-normalized embeddings as numpy array
        """
        processed_images = []

        for img in images:
            if isinstance(img, bytes):
                # Convert bytes to PIL Image
                pil_img = Image.open(io.BytesIO(img))
            elif isinstance(img, str):
                # Assume base64 encoded
                img_bytes = base64.b64decode(img)
                pil_img = Image.open(io.BytesIO(img_bytes))
            elif isinstance(img, Image.Image):
                pil_img = img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")

            # Convert to RGB if necessary
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')

            processed_images.append(pil_img)

        # Encode images using Jina v4's encode_image method
        with torch.no_grad():
            embeddings = self.model.encode_image(
                images=processed_images,
                task="retrieval"
            )

            # Handle CUDA tensors properly
            if torch.is_tensor(embeddings):
                if embeddings.is_cuda:
                    embeddings = embeddings.cpu()
                embeddings = embeddings.numpy()
            elif hasattr(embeddings, 'detach'):
                embeddings = embeddings.detach()
                if hasattr(embeddings, 'is_cuda') and embeddings.is_cuda:
                    embeddings = embeddings.cpu()
                embeddings = embeddings.numpy()
            elif not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def embed_multimodal(self, pairs: list[dict[str, str | list[bytes | Image.Image | str]]]) -> np.ndarray:
        """
        Create unified embeddings for text+image pairs.
        
        Args:
            pairs: List of dicts with 'text' and optional 'images' keys
            
        Returns:
            L2-normalized multimodal embeddings
        """
        embeddings_list = []

        for pair in pairs:
            text: str = str(pair.get('text', ''))
            images_raw = pair.get('images', [])
            images: list[bytes | Image.Image | str] = []
            if images_raw:
                if isinstance(images_raw, list):
                    images = images_raw
                else:
                    images = [images_raw]

            if not text and not images:
                # Empty pair, return zero vector with correct dimensionality
                embeddings_list.append(np.zeros(self.EMBEDDING_DIM, dtype=np.float32))
                continue

            # Use late fusion as Jina v4 doesn't have true multimodal yet
            components = []
            weights = []

            if text:
                text_emb = self.embed_texts([text])[0]
                components.append(text_emb)
                weights.append(0.7)  # Default text weight

            if images:
                img_embs = self.embed_images(images)
                # Average multiple images
                img_emb = np.mean(img_embs, axis=0)
                components.append(img_emb)
                weights.append(0.3)  # Default image weight

            # Weighted combination
            if len(components) == 1:
                combined = components[0]
            else:
                weights = np.array(weights) / np.sum(weights)  # Normalize weights
                combined = np.sum([w * c for w, c in zip(weights, components, strict=False)], axis=0)

            # L2 normalize
            norm = np.linalg.norm(combined)
            combined = combined / (norm + 1e-8)

            embeddings_list.append(combined)

        return np.array(embeddings_list)

    def embed_with_late_chunking(self,
                                 text: str,
                                 task: str = "retrieval.passage") -> list[ChunkWithEmbedding]:
        """
        Implement PROPER late chunking as mandated by CLAUDE.md.

        For now, we use a hybrid approach:
        1. Split text into overlapping windows that fit within model context
        2. Each window gets FULL contextual encoding (not just the chunk)
        3. This ensures chunks have awareness of surrounding context

        While not perfect late chunking (which requires hidden states),
        this is much better than naive chunking and works with Jina's API.

        Args:
            text: Full document text to process
            task: Task type (retrieval, text-matching, separation, classification)

        Returns:
            List of ChunkWithEmbedding objects with context-aware embeddings
        """
        if not text:
            return []

        # Encode the full document once to obtain contextual token embeddings
        token_embeddings, metadata = self.encode_full_document(text, task)

        if token_embeddings.numel() == 0:
            # Fallback: model returned no embeddings (extremely short text)
            jina_task = 'retrieval' if 'retrieval' in task else task
            embedding = self.embed_texts([text], task=jina_task)[0]
            embedding = embedding if isinstance(embedding, np.ndarray) else np.asarray(embedding)
            if embedding.size == 0:
                embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            return [ChunkWithEmbedding(
                text=text,
                embedding=embedding,
                start_char=0,
                end_char=len(text),
                start_token=0,
                end_token=metadata.get('num_tokens', len(text) // 4),
                chunk_index=0,
                total_chunks=1,
                context_window_used=metadata.get('num_tokens', len(text) // 4)
            )]

        # Ensure embeddings are on CPU for downstream numpy ops
        if hasattr(token_embeddings, 'is_cuda') and token_embeddings.is_cuda:
            token_embeddings = token_embeddings.cpu()

        # Run the second stage of late chunking directly on the cached embeddings
        chunks = self.embed_chunks_from_tokens(
            token_embeddings=token_embeddings,
            metadata=metadata,
            text=text,
            chunk_size_tokens=self.chunk_size_tokens,
            chunk_overlap_tokens=self.chunk_overlap_tokens
        )

        return chunks

    def embed_batch_with_late_chunking(self,
                                       texts: list[str],
                                       task: str = "retrieval.passage") -> list[list[ChunkWithEmbedding]]:
        """
        Batch version of embed_with_late_chunking for GPU efficiency.

        Processes multiple documents with proper late chunking:
        1. Each document is fully encoded to get contextualized representations
        2. Then chunks are created from the contextualized hidden states
        3. This preserves full document context in every chunk

        Args:
            texts: List of full document texts to process
            task: Task type (retrieval, text-matching, separation, classification)

        Returns:
            List of lists - one ChunkWithEmbedding list per input document
        """
        if not texts:
            return []

        all_results: list[list[ChunkWithEmbedding]] = []

        # Process each document with proper late chunking
        # We process individually to maintain full context for each document
        for text in texts:
            if not text:
                all_results.append([])
                continue

            # Use the proper late chunking implementation
            chunks = self.embed_with_late_chunking(text, task=task)
            all_results.append(chunks)

        return all_results


    def _prepare_simple_chunks(self, text: str) -> list[dict]:
        """
        Simple chunking method that always works.
        Creates chunks based on chunk_size_tokens with overlap.
        """
        chunks = []

        # Character-based chunking (rough token estimate)
        chars_per_token = 4
        chunk_size_chars = self.chunk_size_tokens * chars_per_token
        overlap_chars = self.chunk_overlap_tokens * chars_per_token

        start_char = 0
        chunk_index = 0

        while start_char < len(text):
            # Define chunk boundaries
            end_char = min(start_char + chunk_size_chars, len(text))

            # Try to break at sentence boundary if possible
            if end_char < len(text):
                # Look for sentence end near boundary
                search_start = max(start_char, end_char - 100)
                sentence_end = text.find('. ', search_start, end_char)
                if sentence_end != -1:
                    end_char = sentence_end + 2

            chunk_text = text[start_char:end_char]

            chunks.append({
                'text': chunk_text,
                'start_char': start_char,
                'end_char': end_char,
                'start_token': start_char // chars_per_token,
                'end_token': end_char // chars_per_token,
                'chunk_index': chunk_index,
                'total_chunks': 0,  # Will be updated
                'context_size': len(chunk_text) // chars_per_token
            })

            # Move to next chunk with overlap
            if end_char >= len(text):
                break

            start_char = end_char - overlap_chars
            chunk_index += 1

        # Update total chunks
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total

        return chunks

    def _prepare_chunks_for_batch(self, text: str, doc_idx: int) -> list[dict]:
        """
        Prepare chunks and context windows for a single document in batch processing.
        
        Returns list of dictionaries with chunk and context information.
        """
        # Estimate chunk size in characters (rough: ~4 chars per token)
        chunk_size_chars = self.chunk_size_tokens * 4

        chunks = []
        chunk_index = 0
        start_char = 0

        while start_char < len(text):
            # Define chunk boundaries
            end_char = min(start_char + chunk_size_chars, len(text))
            chunk_text = text[start_char:end_char]

            # Define context window (chunk + surrounding text)
            context_start = max(0, start_char - chunk_size_chars)
            context_end = min(len(text), end_char + chunk_size_chars)
            context_text = text[context_start:context_end]

            chunk_info = {
                'chunk_text': chunk_text,
                'context_text': context_text,
                'start_char': start_char,
                'end_char': end_char,
                'start_token': start_char // 4,  # Rough estimate
                'end_token': end_char // 4,
                'chunk_index': chunk_index,
                'total_chunks': 0,  # Will be updated later
                'context_window_used': len(context_text) // 4  # Rough token estimate
            }

            chunks.append(chunk_info)

            # Move to next chunk with overlap
            if end_char >= len(text):
                break
            start_char = end_char - (self.chunk_overlap_tokens * 4)  # Convert overlap to chars
            chunk_index += 1

        # Update total chunks count
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)

        logger.debug(f"Prepared {len(chunks)} chunks for document {doc_idx}")
        return chunks

    def _chunk_with_context_windows(self,
                                    text: str,
                                    task: str = "retrieval.passage") -> list[ChunkWithEmbedding]:
        """
        DEPRECATED: This method has O(N^2) complexity due to redundant encoding.

        Use embed_with_late_chunking() instead which properly chunks without
        redundant model calls. This method is kept only for backward compatibility
        and will be removed in future versions.
        """
        logger.warning("_chunk_with_context_windows is deprecated due to O(N^2) complexity. "
                      "Using embed_with_late_chunking instead for better performance.")

        # Redirect to the efficient implementation
        return self.embed_with_late_chunking(text, task)

    def encode_full_document(self,
                           text: str,
                           task: str = "retrieval.passage") -> tuple[torch.Tensor, dict]:
        """
        Encode a full document to get token-level embeddings (first step of late chunking).
        
        This is the critical first step of proper late chunking:
        1. Process the entire document through the transformer
        2. Get contextualized token embeddings for the whole document
        3. Return these for subsequent chunking with context preservation
        
        Args:
            text: Full document text
            task: Task type (retrieval, code, etc.)
            
        Returns:
            Tuple of (token_embeddings, metadata_dict)
            where metadata_dict contains token offsets and other info
        """
        if not text:
            return torch.empty(0, self.EMBEDDING_DIM), {}

        # Check if truncation will be needed
        estimated_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        if estimated_tokens > self.MAX_TOKENS:
            logger.warning(
                f"Document will be truncated from ~{estimated_tokens} to {self.MAX_TOKENS} tokens. "
                f"Consider using process_long_document() for documents > 32k tokens."
            )

        # Tokenize the full document
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,  # Truncate to MAX_TOKENS if needed
            max_length=self.MAX_TOKENS,
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )

        # Move to device
        input_ids = tokens['input_ids'].to(self.model.device)
        attention_mask = tokens['attention_mask'].to(self.model.device)
        offset_mapping = tokens['offset_mapping'][0].cpu().numpy()
        special_tokens_mask = tokens['special_tokens_mask'][0].cpu().numpy()

        with torch.no_grad():
            # Map task to Jina v4 task labels
            # Map task to Jina v4 LoRA adapter names.
            # Valid adapters: 'retrieval', 'text-matching', 'code'
            task_mapping = {
                'retrieval.passage': 'retrieval',
                'retrieval.query': 'retrieval',
                'retrieval': 'retrieval',
                'text-matching': 'text-matching',
                'code': 'code',
            }
            task_label = task_mapping.get(task, 'retrieval')

            # For Jina v4, we need to pass task_label to the model
            # Access the underlying transformer and add task_label
            if hasattr(self.model, 'model'):
                # Call the underlying model with task_label
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_label=task_label,
                    output_hidden_states=True
                )
            else:
                # Fallback to direct model call
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_label=task_label,
                    output_hidden_states=True
                )

            # Get the last hidden state (contextualized token embeddings)
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                token_embeddings = outputs.last_hidden_state[0]  # Shape: [seq_len, hidden_dim]
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # Some models put it in hidden_states
                token_embeddings = outputs.hidden_states[-1][0]  # Last layer, first batch
            elif hasattr(outputs, 'vlm_last_hidden_states') and outputs.vlm_last_hidden_states is not None:
                # Jina v4 specific: use vlm_last_hidden_states for token-level embeddings
                token_embeddings = outputs.vlm_last_hidden_states[0]  # Shape: [seq_len, hidden_dim]
            elif hasattr(outputs, 'multi_vec_emb') and outputs.multi_vec_emb is not None:
                # Jina v4: multi_vec_emb contains token embeddings before pooling
                token_embeddings = outputs.multi_vec_emb[0]
            elif isinstance(outputs, tuple):
                # For tuple outputs, first element is usually the embeddings
                token_embeddings = outputs[0][0] if len(outputs[0].shape) > 2 else outputs[0]
            else:
                # For custom output types like JinaEmbeddingsV4ModelOutput
                # Try to get the first available tensor attribute
                found_embeddings = False
                for attr_name in ['embeddings', 'last_hidden_state', 'hidden_states', 'pooler_output', 'single_vec_emb']:
                    if hasattr(outputs, attr_name):
                        attr_value = getattr(outputs, attr_name)
                        if attr_value is not None:
                            if isinstance(attr_value, (list, tuple)) and len(attr_value) > 0:
                                token_embeddings = attr_value[-1][0] if hasattr(attr_value[-1], 'shape') else attr_value[0]
                            else:
                                token_embeddings = attr_value[0] if hasattr(attr_value, 'shape') and len(attr_value.shape) > 2 else attr_value
                            found_embeddings = True
                            break

                if not found_embeddings:
                    # Last resort: raise a more informative error
                    available_attrs = [attr for attr in dir(outputs) if not attr.startswith('_')]
                    raise ValueError(f"Could not extract token embeddings from {type(outputs).__name__}. Available attributes: {available_attrs}")

            # Apply pooling or projection if needed to get to 2048 dims
            # (Jina v4 uses a projection layer for final embeddings)
            if hasattr(self.model, 'encode'):
                # Use Jina's projection mechanism
                # We need to pool to get document embedding first
                pooled = token_embeddings.mean(dim=0, keepdim=True)
                # For now, we'll just use the pooled embeddings directly
                # since we can't pass pre-computed embeddings to encode
                # Use this as a reference for projection
            else:
                # Direct use of token embeddings
                pass

        metadata = {
            'offset_mapping': offset_mapping,
            'num_tokens': len(input_ids[0]),
            'text_length': len(text),
            'task': task,
            'special_tokens_mask': special_tokens_mask,
        }

        return token_embeddings, metadata

    def embed_chunks_from_tokens(self,
                                 token_embeddings: torch.Tensor,
                                 metadata: dict,
                                 text: str,
                                 chunk_size_tokens: int | None = None,
                                 chunk_overlap_tokens: int | None = None) -> list[ChunkWithEmbedding]:
        """
        Create chunks with embeddings from pre-computed token embeddings (second step).
        
        This is the second step of proper late chunking:
        1. Take the contextualized token embeddings from step 1
        2. Create chunks by slicing the token embeddings
        3. Pool each chunk's tokens to get chunk embedding
        4. Each chunk embedding preserves full document context
        
        Args:
            token_embeddings: Token-level embeddings from encode_full_document
            metadata: Metadata dict from encode_full_document
            text: Original text for creating chunk text
            chunk_size_tokens: Override default chunk size
            chunk_overlap_tokens: Override default overlap
            
        Returns:
            List of ChunkWithEmbedding objects with context-aware embeddings
        """
        chunk_size = chunk_size_tokens or self.chunk_size_tokens
        overlap = chunk_overlap_tokens or self.chunk_overlap_tokens

        offset_mapping = metadata['offset_mapping']
        num_tokens = metadata['num_tokens']
        special_mask = metadata.get('special_tokens_mask')

        chunks = []
        chunk_index = 0
        start_token = 0

        while start_token < num_tokens:
            # Define chunk token boundaries
            end_token = min(start_token + chunk_size, num_tokens)

            # Get character boundaries from offset mapping
            valid_start = start_token
            if special_mask is not None:
                while valid_start < end_token and special_mask[valid_start] == 1:
                    valid_start += 1
            if valid_start >= end_token:
                if end_token >= num_tokens:
                    break
                start_token = end_token
                continue

            valid_end = end_token - 1
            if special_mask is not None:
                while valid_end > valid_start and special_mask[valid_end] == 1:
                    valid_end -= 1

            start_offset = offset_mapping[valid_start]
            end_offset = offset_mapping[valid_end]

            # Handle if these are still tensors
            if hasattr(start_offset, 'cpu'):
                start_offset = start_offset.cpu().numpy()
            if hasattr(end_offset, 'cpu'):
                end_offset = end_offset.cpu().numpy()

            start_char = int(start_offset[0])
            end_char = int(end_offset[1])

            # Extract chunk text
            chunk_text = text[start_char:end_char]

            # Get chunk embedding by pooling token embeddings
            chunk_token_embeddings = token_embeddings[valid_start:valid_end + 1]
            if chunk_token_embeddings.shape[0] == 0:
                if end_token >= num_tokens:
                    break
                start_token = end_token
                continue

            # Mean pooling over tokens in the chunk
            chunk_embedding = chunk_token_embeddings.mean(dim=0)

            # Ensure tensor is on CPU before numpy conversion
            if hasattr(chunk_embedding, 'is_cuda') and chunk_embedding.is_cuda:
                chunk_embedding = chunk_embedding.cpu()

            # Convert to numpy and normalize
            chunk_embedding_np = chunk_embedding.numpy().astype(np.float32, copy=False)
            norm = np.linalg.norm(chunk_embedding_np)
            if norm > 0:
                chunk_embedding_np = chunk_embedding_np / norm

            # Create ChunkWithEmbedding object
            chunks.append(ChunkWithEmbedding(
                text=chunk_text,
                embedding=chunk_embedding_np,
                start_char=start_char,
                end_char=end_char,
                start_token=valid_start,
                end_token=valid_end + 1,
                chunk_index=chunk_index,
                total_chunks=0,  # Will update after loop
                context_window_used=num_tokens  # Full document context
            ))

            # Move to next chunk with overlap
            if end_token >= num_tokens:
                break
            start_token = max(end_token - overlap, 0)
            chunk_index += 1

        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        logger.debug(f"Created {len(chunks)} chunks from {num_tokens} tokens with full context")
        return chunks

    def process_long_document(self,
                             text: str,
                             task: str = "retrieval.passage") -> list[ChunkWithEmbedding]:
        """
        Process a document that may exceed 32k tokens.
        
        For documents longer than 32k tokens, we process in windows with
        overlap to maintain some context across boundaries.
        
        Args:
            text: Document text (can be very long)
            task: Task type
            
        Returns:
            List of all chunks with embeddings
        """
        # Quick token count estimate (rough: ~4 chars per token)
        estimated_tokens = len(text) // 4

        logger.debug(f"process_long_document: text length={len(text)}, estimated tokens={estimated_tokens}")

        # For Jina v4, we use context window approach which already provides
        # excellent context preservation through overlapping windows
        # The model's encode_text method handles the embedding internally

        if estimated_tokens <= self.MAX_TOKENS:
            # Document fits in model's context window
            return self._chunk_with_context_windows(text, task)

        # Process very long documents in overlapping windows
        logger.info(f"Document too long (~{estimated_tokens} tokens), processing in windows")

        all_chunks = []
        window_size_chars = self.MAX_TOKENS * 4  # Rough estimate
        window_overlap_chars = 1000 * 4  # 1000 token overlap

        start = 0
        window_index = 0

        while start < len(text):
            end = min(start + window_size_chars, len(text))
            window_text = text[start:end]

            # Process this window with context windows approach
            window_chunks = self._chunk_with_context_windows(window_text, task)

            # Adjust character positions to be relative to full document
            for chunk in window_chunks:
                chunk.start_char += start
                chunk.end_char += start
                chunk.start_token += start // 4
                chunk.end_token += start // 4

            all_chunks.extend(window_chunks)

            if end >= len(text):
                break

            start = end - window_overlap_chars
            window_index += 1

        # Re-index chunks
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(all_chunks)

        logger.info(f"Processed {window_index + 1} windows, total {len(all_chunks)} chunks")

        return all_chunks
