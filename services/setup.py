"""Build service wheels from canonical protos, including standalone sdists."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent
PROTO_FILES = (
    "persephone/common/common.proto",
    "persephone/extraction/extraction.proto",
    "persephone/embedding/embedding.proto",
    "hades/training/training.proto",
)


def proto_root():
    # A source distribution carries its own copy; a checkout uses canonical files.
    source = ROOT / "_proto" if (ROOT / "_proto").is_dir() else ROOT.parent / "proto"
    for name in PROTO_FILES:
        if not (source / name).is_file():
            raise RuntimeError(f"Missing required protobuf source: {source / name}")
    return source


def generate(destination):
    from grpc_tools import protoc

    source = proto_root()
    destination.mkdir(parents=True, exist_ok=True)
    args = ["grpc_tools.protoc", f"-I{source}", f"--python_out={destination}",
            f"--grpc_python_out={destination}"]
    if protoc.main(args + [str(source / name) for name in PROTO_FILES]):
        raise RuntimeError("Service protobuf generation failed")
    for directory in [destination, *(p for p in destination.rglob("*") if p.is_dir())]:
        (directory / "__init__.py").touch()


class BuildServices(build_py):
    def run(self):
        super().run()
        # Editable services import their sibling generated directory directly.
        target = ROOT if self.editable_mode else Path(self.build_lib)
        generate(target / "generated")


class SourceServices(sdist):
    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        source = proto_root()
        for name in PROTO_FILES:
            target = Path(base_dir) / "_proto" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)


setup(cmdclass={"build_py": BuildServices, "sdist": SourceServices})
