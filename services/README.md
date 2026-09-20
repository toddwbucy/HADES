# Python service distribution

The `hades-services` wheel includes extraction, embedding, training, generated
RPC bindings, and the existing WeaverTools adapter with its `schema.yaml` resource.
The adapter remains project-specific; see `adapters/weavertools/README.md`.

## Build and install

From a fresh repository checkout, build in a dedicated environment:

```bash
python -m pip wheel --no-deps ./services --wheel-dir /tmp/hades-wheels
python -m pip install /tmp/hades-wheels/hades_services-0.1.0-py3-none-any.whl
```

The isolated build installs the pinned setuptools and gRPC code generator from
`pyproject.toml`. It generates RPC modules directly into wheel build output from
canonical `../proto` sources. Missing proto inputs fail the build. No manual
`make proto-gen` step is needed for wheels. Source archives created with
`python -m build --sdist services` include the same four proto inputs under
`_proto/`, so rebuilding does not depend on the parent repository.

Generated code requires `grpcio>=1.84.0` and `protobuf>=7.35.1,<8`; these floors
match the pinned generator's emitted runtime checks. The generator is a build
and development dependency, not an inference runtime requirement. Runtime ML
libraries remain declared in `pyproject.toml`; model weights are separate assets.

For source development, install `./services[dev]` in editable mode in a dedicated
venv. The editable build generates sibling `services/generated/` bindings;
`make -C services proto-gen PROTO_DIR=../proto PYTHON=/path/to/venv/bin/python`
can regenerate them after protocol edits. Launch modules with `python -m
training.server`, `python -m extraction.server`, or `python -m embedding.http_server`
using explicit service configuration. Installing a wheel does not provision
systemd units or start services.

## Distribution contracts

With the hashed CPU CI dependencies installed, run:

```bash
python scripts/test_python_distribution.py
```

This uses an isolated PEP 517 frontend to resolve declared build dependencies
and build a fresh wheel and standalone source archive in temporary directories,
installs both direct and rebuilt wheels into temporary targets, and imports service
and generated RPC modules from outside the checkout with isolated Python paths.
It checks the adapter schema resource and loads no models or running services.
CI runs this before source-tree contract tests, so editable imports cannot hide
missing wheel files. Use isolated environments for installation tests; do not
install into an active service's virtual environment.

The build command extensions follow the [setuptools extension API](https://setuptools.pypa.io/en/latest/userguide/extension.html).
