"""Trusted local registration and private serving copies, using only synthetic assets."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embedding.registry import MODEL_ID, ModelRegistry
from embedding.snapshot import ArtifactChangedError, remove_private_tree

REVISION = "a" * 40


@pytest.fixture
def source(tmp_path):
    """Build a self-contained synthetic Jina-shaped directory with no executable model."""
    root = tmp_path / "source"
    root.mkdir()
    assets = {
        "config.json": json.dumps({"auto_map": {"AutoModel": "modeling_fixture.Fixture"}}),
        "tokenizer_config.json": "{}", "tokenizer.json": "{}",
        "preprocessor_config.json": "{}", "model.safetensors": "synthetic weights",
        "adapters/adapter_config.json": "{}", "adapters/adapter_model.safetensors": "synthetic adapter",
        "modeling_fixture.py": "# synthetic local model code\n", "modules.json": "[]",
    }
    for name, content in assets.items():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(content)
    return root


@pytest.fixture
def registry(tmp_path):
    """Keep every registry write within a disposable private directory."""
    root = tmp_path / "registry"
    instance = ModelRegistry(root)
    yield instance
    remove_private_tree(root)


def test_registration_persists_private_identity_without_loading(source, registry):
    """A durable registration is an expected identity, not a loaded-model assertion."""
    model = registry.register("local-jina", source, REVISION)
    loaded = ModelRegistry(registry.root).load("local-jina")
    assert model == loaded
    assert model.model_id == MODEL_ID
    assert (registry.root / "records/local-jina.json").stat().st_mode & 0o777 == 0o600
    assert model.artifacts.stat().st_mode & 0o777 == 0o500
    assert all((model.artifacts / file.path).stat().st_mode & 0o777 == 0o400 for file in model.manifest.files)
    assert not list(model.sessions.iterdir())
    assert (source / "model.safetensors").read_text() == "synthetic weights"


def test_relocation_metadata_and_cache_links_do_not_change_identity(source, registry, tmp_path):
    """Hash resolved asset bytes under logical paths and exclude known cache metadata."""
    first = registry.register("first", source, REVISION)
    (source / "README.md").write_text("documentation")
    (source / "results.json").write_text('{"unrelated_evaluation": true}')
    (source / ".cache").mkdir()
    (source / ".cache/download.lock").write_text("not an asset")
    blob = tmp_path / "cached-blob"
    (source / "model.safetensors").rename(blob)
    (source / "model.safetensors").symlink_to(blob)
    second = registry.register("second", source, REVISION)
    assert first.manifest.digest() == second.manifest.digest()
    assert not (second.artifacts / "model.safetensors").is_symlink()
    assert (second.artifacts / "model.safetensors").stat().st_ino != blob.stat().st_ino
    blob.write_text("changed after registration")
    snapshot = second.prepare()
    try:
        snapshot.check()
        assert (snapshot.root / "model.safetensors").read_text() == "synthetic weights"
    finally:
        snapshot.close()


@pytest.mark.parametrize("asset", ["model.safetensors", "tokenizer.json", "config.json",
                                   "adapters/adapter_model.safetensors", "modeling_fixture.py"])
def test_every_runtime_dependency_changes_registered_identity(source, registry, asset):
    """Weights, tokenizer/config, adapter, and model-local code all participate in MD5."""
    original = registry.register("before", source, REVISION)
    with (source / asset).open("a") as stream:
        stream.write("\n")
    changed = registry.register("after", source, REVISION)
    assert original.manifest.digest() != changed.manifest.digest()


@pytest.mark.parametrize("missing", ["model.safetensors", "tokenizer.json", "preprocessor_config.json",
                                     "adapters/adapter_config.json", "adapters/adapter_model.safetensors",
                                     "modeling_fixture.py"])
def test_incomplete_registration_leaves_no_ready_record_or_artifacts(source, registry, missing):
    """Reject an incomplete bundle before publishing its expected identity."""
    (source / missing).unlink()
    with pytest.raises(ValueError):
        registry.register("incomplete", source, REVISION)
    assert not list((registry.root / "records").iterdir())
    assert not list((registry.root / "artifacts").iterdir())


def test_shard_index_requires_every_declared_weight_file(source, registry):
    """Do not treat an index as proof that all referenced shards were included."""
    (source / "model.safetensors").rename(source / "model-00001.safetensors")
    index = source / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "model-00001.safetensors", "b": "missing.safetensors"}}))
    with pytest.raises(ValueError, match="shards"):
        registry.register("missing-shard", source, REVISION)
    index.write_text(json.dumps({"weight_map": {"a": "model-00001.safetensors"}}))
    assert registry.register("complete-shard", source, REVISION).manifest.files


def test_remote_code_references_and_unknown_files_are_refused(source, registry):
    """Registration must not authorize remote auto_map code or silently omit unknown assets."""
    config = source / "config.json"
    config.write_text(json.dumps({"auto_map": {"AutoModel": "other/repo--modeling.Remote"}}))
    with pytest.raises(ValueError, match="remote"):
        registry.register("remote", source, REVISION)
    config.write_text("{}")
    (source / "unknown.blob").write_bytes(b"unclassified")
    with pytest.raises(ValueError, match="unsupported"):
        registry.register("unknown", source, REVISION)


@pytest.mark.parametrize("special", ["directory-link", "fifo"])
def test_special_assets_fail_without_following_or_blocking(source, registry, tmp_path, special):
    """Never traverse linked directories or block while trying to copy a FIFO."""
    if special == "directory-link":
        (source / "linked").symlink_to(tmp_path, target_is_directory=True)
    else:
        os.mkfifo(source / "blocked.bin")
    with pytest.raises(ValueError):
        registry.register("special", source, REVISION)


def test_provider_ids_are_immutable_and_concurrent_registration_has_one_winner(source, registry):
    """Independent registry instances serialize creation and never overwrite an ID."""
    def attempt():
        """Record either the one successful registration or its expected collision."""
        try:
            return ModelRegistry(registry.root).register("shared", source, REVISION)
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: attempt(), range(2)))
    assert sum(value is not None for value in results) == 1
    assert len(list((registry.root / "artifacts").iterdir())) == 1
    with pytest.raises(ValueError, match="already registered"):
        registry.register("shared", source, "b" * 40)


def test_registration_detects_source_changes_and_budget_failure(source, registry, monkeypatch):
    """Partial copies cannot leave a successful record when the source changes or exceeds budget."""
    with pytest.raises(ValueError, match="byte budget"):
        registry.register("oversize", source, REVISION, max_bytes=1)
    import embedding.registry as module
    original = module.copy_assets
    def changed(*args, **kwargs):
        """Inject source drift after copying, before the registration consistency check."""
        result = original(*args, **kwargs)
        (source / "tokenizer.json").write_text('{"changed": true}')
        return result
    monkeypatch.setattr(module, "copy_assets", changed)
    with pytest.raises(ValueError, match="source changed"):
        registry.register("changed", source, REVISION)
    assert not list((registry.root / "records").iterdir())
    assert not list((registry.root / "artifacts").iterdir())


def test_unsafe_records_and_changed_persisted_bytes_are_not_trusted(source, registry):
    """Protected record syntax and a fresh load's byte hashes both gate serving."""
    model = registry.register("safe", source, REVISION)
    record = registry.root / "records/safe.json"
    record.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        registry.load("safe")
    record.chmod(0o600)
    value = json.loads(record.read_text())
    value["manifest_md5"] = "0" * 32
    record.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="MD5 mismatch"):
        registry.load("safe")
    asset = model.artifacts / "model.safetensors"
    asset.chmod(0o600)
    asset.write_bytes(b"different weights")
    asset.chmod(0o400)
    with pytest.raises(ValueError, match="MD5 mismatch"):
        model.prepare()
    assert not list(model.sessions.iterdir())


@pytest.mark.parametrize("mutation", ["write", "replace", "insert", "remove"])
def test_snapshot_seal_detects_all_mutation_classes(source, registry, mutation):
    """Metadata checks catch write-and-restore permissions and namespace changes."""
    model = registry.register("sealed", source, REVISION)
    snapshot = model.prepare()
    root = snapshot.root
    try:
        target = root / "model.safetensors"
        root.chmod(0o700)
        if mutation == "write":
            target.chmod(0o600)
            target.write_text("synthetic weights")  # even equal bytes invalidate this load
            target.chmod(0o400)
        elif mutation == "replace":
            replacement = root / "replacement"
            replacement.write_text("synthetic weights")
            replacement.chmod(0o400)
            replacement.replace(target)
        elif mutation == "insert":
            (root / "extra.json").write_text("{}")
            (root / "extra.json").chmod(0o400)
        else:
            target.unlink()
        root.chmod(0o500)
        with pytest.raises(ArtifactChangedError):
            snapshot.check()
    finally:
        snapshot.close()
    assert not root.exists()


def test_cleanup_never_follows_a_replaced_snapshot_root(tmp_path):
    """A symlink replacing a private root must not cause removal of its target."""
    target = tmp_path / "keep"
    target.mkdir()
    (target / "keep.txt").write_text("preserve")
    root = tmp_path / "replaced"
    root.symlink_to(target, target_is_directory=True)
    remove_private_tree(root)
    assert (target / "keep.txt").read_text() == "preserve"


def test_cli_register_and_inspect_are_offline_and_do_not_claim_a_loaded_model(source, registry):
    """Exercise the actual administrative entry point against a disposable registry."""
    base = [sys.executable, "-m", "embedding.registry", "--registry", str(registry.root)]
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    created = subprocess.run(base + ["register", "--provider", "cli", "--source", str(source),
                                    "--revision", REVISION], env=env, capture_output=True, text=True, check=True)
    inspected = subprocess.run(base + ["inspect", "--provider", "cli"], env=env,
                               capture_output=True, text=True, check=True)
    assert json.loads(created.stdout) == json.loads(inspected.stdout)
    assert json.loads(created.stdout)["loaded_verified"] is False
