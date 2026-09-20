#!/usr/bin/env python3
"""Build fresh wheel/sdist and verify installed imports outside the checkout."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]


def run(args, cwd):
    subprocess.run([sys.executable, *args], cwd=cwd, check=True,
                   env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                   timeout=120)


def verify_wheel(wheel, root):
    target = root / "installed"
    run(["-m", "pip", "install", "--no-cache-dir", "--no-deps", "--no-index", "--target", str(target), str(wheel)], root)
    # -I excludes cwd, PYTHONPATH and user site; the explicit target must win.
    run(["-I", "-c", '''
import importlib, importlib.resources, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
for name in ("training.server", "extraction.server", "embedding.config",
             "embedding.tensors", "adapters.weavertools.extractor"):
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(root), name
from hades.training import training_pb2, training_pb2_grpc
from persephone.embedding import embedding_pb2, embedding_pb2_grpc
from persephone.extraction import extraction_pb2, extraction_pb2_grpc
assert training_pb2.ModelConfig(hidden_dim=8).hidden_dim == 8
schema = importlib.resources.files("adapters.weavertools").joinpath("schema.yaml")
assert "collections:" in schema.read_text()
print("Installed wheel service imports and adapter resource passed")
''', str(target)], root)


def main():
    with tempfile.TemporaryDirectory(prefix="hades-python-package-") as temporary:
        root = Path(temporary)
        source = root / "checkout" / "services"
        shutil.copytree(REPO / "services", source, ignore=shutil.ignore_patterns(
            ".venv", "__pycache__", "*.egg-info", "build", "dist", "generated", "proto"))
        shutil.copytree(REPO / "proto", source.parent / "proto")
        run(["-c", "from setuptools.build_meta import build_wheel, build_sdist; "
             "build_wheel('dist'); build_sdist('dist')"], source)
        direct = root / "direct"
        direct.mkdir()
        verify_wheel(next((source / "dist").glob("*.whl")), direct)
        archive = next((source / "dist").glob("*.tar.gz"))
        standalone = root / "standalone"
        with tarfile.open(archive) as stream:
            stream.extractall(standalone, filter="data")
        unpacked = next(standalone.iterdir())
        # There is no ../proto here: the sdist must carry all of its own inputs.
        run(["-c", "from setuptools.build_meta import build_wheel; build_wheel('dist')"], unpacked)
        rebuilt = root / "rebuilt"
        rebuilt.mkdir()
        verify_wheel(next((unpacked / "dist").glob("*.whl")), rebuilt)
        assert not (source / "generated").exists(), "wheel builds must not generate into source"
        run(["-c", "from setuptools.build_meta import build_editable; build_editable('editable')"], source)
        assert (source / "generated/hades/training/training_pb2.py").is_file()



if __name__ == "__main__":
    main()
