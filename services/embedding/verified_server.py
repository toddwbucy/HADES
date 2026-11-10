"""Entry point for the verified provider service on a private Unix socket."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import stat
import sys
import threading
from pathlib import Path

from fastapi import FastAPI

from .config import EmbeddingConfig
from .jina_v4 import MAX_TOKENS
from .registry import ModelRegistry, RegisteredModel
from .verified_http import create_verified_app
from .verified_jina import VerifiedJinaProvider
from .verified_late import VerifiedJinaLateProvider

logger = logging.getLogger(__name__)

# One process serves exactly one embedding space (#225). The late provider is a distinct
# space, not a superset: it rewrites the pooling identity, refuses conventional planning
# and embeds only queries, so the operator names the space rather than the service
# guessing it.
SPACES: dict[str, type[VerifiedJinaProvider]] = {
    "conventional": VerifiedJinaProvider,
    "late": VerifiedJinaLateProvider,
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Require every selector explicitly; no space or socket is chosen by default."""
    parser = argparse.ArgumentParser(prog="python -m embedding.verified_server",
                                     description=__doc__)
    parser.add_argument("--registry", required=True, type=Path,
                        help="model registry root holding the registered provider")
    parser.add_argument("--provider", required=True, help="registered provider ID")
    parser.add_argument("--space", required=True, choices=sorted(SPACES),
                        help="the one embedding space this process serves")
    parser.add_argument("--uds", required=True, type=Path,
                        help="absolute Unix socket path inside a private 0700 directory")
    return parser.parse_args(argv)


def check_socket_path(uds: Path) -> Path:
    """Refuse a socket location the provider catalog would refuse later.

    This mirrors the Rust catalog's socket checks (persephone/provider_registry/mod.rs,
    in `describe`) so the operator learns about an unsafe parent at service
    start, not at the first /v1/verify from HADES (#225). lstat is used throughout:
    a symlinked ancestor would let another path's owner redirect the socket.
    """
    if not uds.is_absolute():
        raise ValueError("socket path must be absolute")
    owner = os.geteuid()
    parent = uds.parent
    info = os.lstat(parent)
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"socket parent must not be a symlink: {parent}")
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != owner or info.st_mode & 0o7777 != 0o700:
        raise ValueError(f"socket parent must be a 0700 directory owned by this account: {parent}")
    for ancestor in parent.parents:
        info = os.lstat(ancestor)
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"socket ancestors must not be symlinks: {ancestor}")
        if not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, owner):
            raise ValueError("socket ancestors must be directories owned by this account "
                             f"or root: {ancestor}")
        shared = info.st_mode & 0o022
        if shared and not (info.st_uid == 0 and info.st_mode & stat.S_ISVTX):
            raise ValueError("socket ancestors must not be writable by other accounts "
                             f"outside root-owned sticky directories: {ancestor}")
    try:
        existing = os.lstat(uds)
    except FileNotFoundError:
        return uds
    if not stat.S_ISSOCK(existing.st_mode):
        raise ValueError("socket path is occupied by a non-socket file")
    # A socket left by a crashed run cannot be reused by bind; the private parent
    # guarantees it was ours, so removing it is safe.
    os.unlink(uds)
    return uds


def build_provider(registration: RegisteredModel, space: str, config: EmbeddingConfig, *,
                   max_input_tokens: int = MAX_TOKENS) -> VerifiedJinaProvider:
    """Construct the selected space's provider on the configured device, without loading.

    The backend silently falls back to CPU when CUDA is unavailable; a 9 GiB model on CPU
    is not a serving configuration, so that fallback is refused here rather than
    discovered as a hung first request (same rule as scripts/qualify_verified_*.py).
    """
    provider = SPACES[space](registration, device=config.device,
                             use_fp16=config.use_fp16 and config.device.startswith("cuda"),
                             max_input_tokens=max_input_tokens)
    if provider.device != config.device:
        raise ValueError("configured device is unavailable; CPU fallback is refused")
    return provider


def build_app(provider: VerifiedJinaProvider, config: EmbeddingConfig) -> FastAPI:
    """Bind the provider and honor the configured idle timeout.

    create_verified_app builds its own default config (900 s idle unload). After an
    unload every held receipt is invalid, so the operator's HADES_EMBEDDER_IDLE_TIMEOUT
    (0 disables the monitor) must reach the app before its lifespan starts (#225).
    """
    app = create_verified_app(provider)
    app.state.app_state.config.idle_timeout_seconds = config.idle_timeout_seconds
    return app


def serve(app: FastAPI, uds: Path) -> None:
    """Serve the app on the socket until a stop signal ends uvicorn's graceful shutdown."""
    import uvicorn

    # uvicorn's capture_signals restores the previous handlers after a graceful
    # shutdown, then calls raise_signal for each signal it caught. Under the default
    # SIGTERM handler that kills the process before main() unlinks the socket and
    # returns 0 (#225). Installed here, not at the top of main(), so a SIGTERM during
    # the eager model load still stops the process by default. signal.signal works
    # only on the main thread, which is where uvicorn listens for signals too.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda signum, frame: None)
    uvicorn.Server(uvicorn.Config(app, uds=str(uds), log_level="info", access_log=False,
                                  timeout_graceful_shutdown=30)).run()


def main(argv: list[str] | None = None) -> int:
    """Load one registered provider eagerly and serve it until stopped.

    Every refusal happens before the socket is bound, so a misconfigured unit fails
    its start instead of listening and failing the first verify. The model loads
    before serving: a client that sees the socket can verify at once, and a load
    failure surfaces in the unit's start rather than in a request.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        # The registered loader refuses otherwise (jina_v4.py); the libraries read
        # these at import, so they cannot be set from inside this process.
        print("verified server requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1",
              file=sys.stderr)
        return 1
    if not args.registry.is_dir():
        # ModelRegistry creates a missing root; a typo must not create an empty registry.
        print(f"verified server: registry root does not exist: {args.registry}",
              file=sys.stderr)
        return 1
    try:
        uds = check_socket_path(args.uds)
        config = EmbeddingConfig.from_env()
        registration = ModelRegistry(args.registry).load(args.provider)
        provider = build_provider(registration, args.space, config)
        description = provider.describe()
        # Identity, never paths: journal output must not leak private locations.
        logger.info("Verified provider ready (model=%s, space=%s, serving_generation=%s)",
                    provider.model_name, provider.space,
                    description["receipt"]["serving_generation"])
        app = build_app(provider, config)
    except (ValueError, OSError) as exc:
        print(f"verified server: {exc}", file=sys.stderr)
        return 1

    try:
        serve(app, uds)
    finally:
        uds.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
