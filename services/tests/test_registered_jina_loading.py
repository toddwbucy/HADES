"""Exercise the production Jina load/inference lifecycle with CPU-only loader doubles."""

from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
import threading
import types

import numpy as np
import pytest
import torch

from test_model_registration import REVISION, registry, source  # noqa: F401
from embedding.registry import MODEL_ID
from embedding.snapshot import ArtifactChangedError


@pytest.fixture
def backend(monkeypatch):
    """Stub only model/tokenizer acquisition; keep real Jina guards and pooling code."""
    state = types.SimpleNamespace(loads=[], calls=0, load_hook=None, infer_hook=None)

    class Encoded(dict):
        """Provide the tokenizer batch's device-transfer interface on CPU."""
        def to(self, device):
            """Return already-CPU synthetic tensors."""
            assert device == "cpu"
            return self

    class Tokenizer:
        """Map synthetic text to character tokens with exact offsets."""
        def __call__(self, texts, **kwargs):
            """Supply either a token count or a late-chunk tokenizer batch."""
            if isinstance(texts, str):
                return {"input_ids": list(range(len(texts)))}
            length = len(texts[0])
            return Encoded(input_ids=torch.arange(length).reshape(1, length),
                           attention_mask=torch.ones((1, length)),
                           offset_mapping=torch.tensor([[(i, i + 1) for i in range(length)]]))

    class Model:
        """Return deterministic CPU vectors without loading any model weights."""
        def to(self, device):
            """Allow simulated CUDA loading without allocating GPU tensors."""
            return self

        def requires_grad_(self, enabled):
            """Assert the real loader disables gradient tracking."""
            assert enabled is False

        def parameters(self):
            """Supply the dtype used by the real loader's status log."""
            return iter([torch.zeros(1)])

        def encode_text(self, batch, **kwargs):
            """Count conventional inference and allow a deterministic race injection."""
            state.calls += 1
            if state.infer_hook:
                state.infer_hook()
            return np.ones((len(batch), 2048), dtype=np.float32)

        def __call__(self, **kwargs):
            """Exercise actual late-chunk pooling over synthetic hidden states."""
            state.calls += 1
            if state.infer_hook:
                state.infer_hook()
            length = kwargs["attention_mask"].shape[1]
            return types.SimpleNamespace(vlm_last_hidden_states=torch.ones((1, length, 2048)))

    def tokenizer_loader(path, **kwargs):
        """Record the exact tokenizer source and offline loading arguments."""
        state.loads.append(("tokenizer", path, kwargs))
        if state.load_hook:
            state.load_hook("tokenizer", Path(path))
        return Tokenizer()

    def model_loader(path, **kwargs):
        """Record the exact model source and allow a load-boundary failure."""
        state.loads.append(("model", path, kwargs))
        if state.load_hook:
            state.load_hook("model", Path(path))
        return Model()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=tokenizer_loader)
    transformers.AutoModel = types.SimpleNamespace(from_pretrained=model_loader)
    hub = types.ModuleType("transformers.utils.hub")
    hub.is_offline_mode = lambda: True
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", hub)
    import huggingface_hub.constants as constants
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    name = "embedding._registered_loader_test"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "embedding/jina_v4.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, state, hub, constants


def make_embedder(backend, registry, source):
    """Construct a registered CPU backend without activating HTTP configuration."""
    registration = registry.register("jina", source, REVISION)
    return backend[0].JinaV4Embedder(model_name=MODEL_ID, device="cpu", registration=registration)


def mutate(root):
    """Simulate an accidental write followed by restoring the read-only mode."""
    path = root / "model.safetensors"
    path.chmod(0o600)
    path.write_bytes(b"changed private copy")
    path.chmod(0o400)


def test_registered_load_uses_private_files_and_reload_gets_fresh_generation(backend, registry, source):
    """Only successful loads publish provenance, and unload discards it with the copy."""
    embedder = make_embedder(backend, registry, source)
    assert embedder.loaded_artifact is None
    try:
        vectors = embedder.embed_texts(["synthetic"])
        assert vectors.shape == (1, 2048)
        first = embedder.loaded_artifact
        first_path = Path(backend[1].loads[0][1])
        assert first_path != source and first_path != embedder._registration.artifacts
        assert first["manifest_md5"] == embedder._registration.manifest.digest()
        assert "space_id" not in first
        for kind, path, options in backend[1].loads:
            assert Path(path) == first_path
            assert options["local_files_only"] is True
            assert options["trust_remote_code"] is True
            if kind == "model":
                assert options["use_safetensors"] is True
        embedder.unload()
        assert not first_path.exists() and embedder.loaded_artifact is None
        embedder.embed_texts(["synthetic"])
        second = embedder.loaded_artifact
        assert first["manifest_md5"] == second["manifest_md5"]
        assert first["serving_generation"] != second["serving_generation"]
        assert backend[1].loads[-1][1] != str(first_path)
    finally:
        embedder.unload()


@pytest.mark.parametrize("point", ["tokenizer", "model", "mutation"])
def test_partial_or_mutated_load_never_publishes_identity(backend, registry, source, point):
    """Failure after either acquisition step releases its private copy and partial state."""
    embedder = make_embedder(backend, registry, source)
    def fail(kind, root):
        """Inject a loader exception or change the private weights during loading."""
        if kind == point:
            raise RuntimeError("synthetic loader failure")
        if point == "mutation" and kind == "model":
            mutate(root)
    backend[1].load_hook = fail
    with pytest.raises((RuntimeError, ArtifactChangedError)):
        embedder.embed_texts(["synthetic"])
    assert embedder.loaded_artifact is None
    assert not embedder.is_loaded and embedder._tokenizer is None
    assert not list(embedder._registration.sessions.iterdir())


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("during", [False, True])
def test_changed_snapshot_cannot_return_accepted_vectors(backend, registry, source, late, during):
    """Both inference paths check before admission and again before returning vectors."""
    embedder = make_embedder(backend, registry, source)
    embedder.model
    root = embedder._snapshot.root
    if during:
        backend[1].infer_hook = lambda: mutate(root)
    else:
        mutate(root)
    with pytest.raises(ArtifactChangedError):
        if late:
            embedder.embed_late_chunked("synthetic", chunk_size_tokens=3, overlap_tokens=0)
        else:
            embedder.embed_texts(["synthetic"])
    assert backend[1].calls == (1 if during else 0)
    assert embedder.loaded_artifact is None and not embedder.is_loaded
    assert not root.exists()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_rejected_snapshot_can_reload_without_reserved_cuda_memory(backend, registry, source, monkeypatch, device):
    """Invalidation can recover without manual unload; CPU and loaded accesses skip cleanup."""
    module = backend[0]
    registration = registry.register("reload", source, REVISION)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(module, "MIN_FREE_VRAM_MIB", 0)
    events = []
    retained = {"cached": False, "collected": False}

    def collect():
        """Model references held in discarded traceback cycles become collectible."""
        assert device.startswith("cuda")
        retained["collected"] = True
        events.append("collect")

    def empty_cache():
        """Only release reserved blocks after collecting discarded references."""
        assert device.startswith("cuda") and retained["collected"]
        retained["cached"] = False
        events.append("empty_cache")

    def mem_get_info(ordinal):
        """Expose insufficient free VRAM until the simulated allocator is cleared."""
        assert ordinal == 0
        events.append("preflight")
        return (7 if retained["cached"] else 16) * 1024**3, 16 * 1024**3

    monkeypatch.setattr(module.gc, "collect", collect)
    monkeypatch.setattr(module.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(module.torch.cuda, "mem_get_info", mem_get_info)
    embedder = module.JinaV4Embedder(device=device, registration=registration)
    try:
        embedder.model
        first = embedder.loaded_artifact
        mutate(embedder._snapshot.root)
        with pytest.raises(ArtifactChangedError):
            embedder.model
        assert not embedder.is_loaded
        retained.update(cached=True, collected=False)
        events.clear()

        embedder.model
        assert embedder.loaded_artifact["serving_generation"] != first["serving_generation"]
        assert events == (["collect", "empty_cache", "preflight"] if device.startswith("cuda") else [])
        events.clear()
        embedder.model
        with embedder._artifact_lock, embedder._load_lock:
            embedder._load_unlocked()
        assert events == []
    finally:
        # Cleanup uses real CPU collection without exercising the CUDA double.
        embedder._device = "cpu"
        monkeypatch.undo()
        embedder.unload()


def test_registered_late_chunking_retains_existing_pooling_behavior(backend, registry, source):
    """The snapshot guard preserves real late-chunk spans and vector shapes."""
    embedder = make_embedder(backend, registry, source)
    try:
        vectors, spans = embedder.embed_late_chunked("abcdef", chunk_size_tokens=3, overlap_tokens=0)
        assert vectors.shape == (2, 2048)
        assert spans == [(0, 3, 0, 3), (3, 6, 3, 6)]
        assert embedder.loaded_artifact is not None
    finally:
        embedder.unload()


def test_unload_waits_for_complete_registered_inference(backend, registry, source):
    """An unload cannot delete files or change generations between inference batches."""
    embedder = make_embedder(backend, registry, source)
    entered = threading.Event()
    release = threading.Event()
    unloading = threading.Event()
    def inference():
        """Hold the first forward pass while an unload tries to acquire ownership."""
        entered.set()
        assert release.wait(5)
    def unload():
        """Signal the unload attempt before waiting for the artifact lifecycle lock."""
        unloading.set()
        embedder.unload()
    backend[1].infer_hook = inference
    with ThreadPoolExecutor(max_workers=2) as executor:
        running = executor.submit(embedder.embed_texts, ["one", "two"], batch_size=1)
        try:
            assert entered.wait(5)
            stopping = executor.submit(unload)
            assert unloading.wait(5)
            assert not stopping.done()
            assert embedder._snapshot.root.exists()
        finally:
            release.set()
        assert running.result(timeout=5).shape == (2, 2048)
        stopping.result(timeout=5)
    assert not embedder.is_loaded and embedder.loaded_artifact is None


def test_registered_constructor_refuses_online_runtime_and_wrong_model(backend, registry, source, monkeypatch):
    """Never infer offline settings from environment strings after libraries initialized."""
    registration = registry.register("offline", source, REVISION)
    module, _, hub, constants = backend
    with pytest.raises(ValueError, match="configured model"):
        module.JinaV4Embedder(device="cpu", model_name="other", registration=registration)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    with pytest.raises(ValueError, match="offline|OFFLINE"):
        module.JinaV4Embedder(device="cpu", registration=registration)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(hub, "is_offline_mode", lambda: False)
    with pytest.raises(ValueError, match="offline|OFFLINE"):
        module.JinaV4Embedder(device="cpu", registration=registration)


def test_unregistered_loader_keeps_existing_configuration_and_no_verified_claim(backend):
    """The inert opt-in boundary leaves today's HTTP-created backend behavior intact."""
    embedder = backend[0].JinaV4Embedder(device="cpu", model_name="legacy/fixture")
    try:
        assert embedder.embed_texts(["synthetic"]).shape == (1, 2048)
        assert embedder.loaded_artifact is None
        for _, path, options in backend[1].loads:
            assert path == "legacy/fixture"
            assert "local_files_only" not in options
    finally:
        embedder.unload()
