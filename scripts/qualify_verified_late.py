#!/usr/bin/env python3
"""Qualify contextual Jina pooling through Rust HTTP with already approved local assets.

Requires the installed ML environment and both Hugging Face offline flags. Uses
a private Unix socket; optional database checks use the strict disposable runner.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import warnings

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "services"))


async def rust_roundtrip(provider, context, output):
    """Own the private transport until the independent tokenizer recount completes."""
    from embedding.late_qualification import recheck
    from qualification_transport import run_rust_contract
    return await run_rust_contract(
        provider, context, output, repo=REPO, target="verified_late_provider",
        test="registered_contextual_transport", environment="HADES_VERIFIED_LATE",
        recheck=lambda p, c, result: recheck(p, c["planning_text"], result["planning"]),
    )


def main(argv=None):
    """Keep success/failure evidence in a new directory, including implementation identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    server = parser.add_mutually_exclusive_group()
    server.add_argument("--arangod", type=Path, help="also qualify file ingestion/setup/search on a private server")
    server.add_argument("--docker", action="store_true", help="also qualify database flow using the pinned isolated image")
    args = parser.parse_args(argv)
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        parser.error("set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before starting")
    if args.device != "cpu" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("device must be cpu or cuda:<ordinal>")
    if args.arangod and not args.arangod.is_file():
        parser.error("arangod must name an existing binary")
    os.umask(0o077)
    output = args.output_dir.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {"protocol": "hades-verified-late-qualification-v1", "passed": False,
              "started_at": datetime.now(timezone.utc).isoformat(), "stage": "preflight"}
    provider = None
    interrupted_run = False
    try:
        import torch
        from embedding.identity import ProviderReceipt
        from embedding.qualification import load_cases, verify_adapters, tokenizer_observation, require
        from embedding.registry import ModelRegistry
        from embedding.verified_late import VerifiedJinaLateProvider
        from embedding.verified_jina import ReceiptMismatchError, _versions
        if args.device.startswith("cuda:") and (
            not torch.cuda.is_available() or int(args.device[5:]) >= torch.cuda.device_count()
        ):
            raise RuntimeError("requested CUDA device is unavailable; CPU fallback is refused")
        cases, digest = load_cases(REPO / "services/tests/fixtures/jina_qualification.json")
        report["suite_sha256"] = digest
        report["implementation_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        report["implementation_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=REPO, text=True).strip())
        registration = ModelRegistry(args.registry).load(args.provider)
        report["manifest"] = json.loads(registration.manifest.canonical_bytes())
        report["runtime"] = {"device": args.device, "libraries": _versions(), "cuda": torch.version.cuda}
        provider = VerifiedJinaLateProvider(registration, device=args.device,
            use_fp16=args.device.startswith("cuda:"), max_input_tokens=1024)
        require(provider._backend.device == args.device, "provider changed device")
        warnings.filterwarnings("error", message="Found missing adapter keys.*")
        report["stage"] = "provider"
        if args.device.startswith("cuda:"):
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            report["runtime"]["gpu"] = torch.cuda.get_device_name(args.device)
        report["description"] = provider.describe()
        report["adapter_tensors_verified"] = verify_adapters(provider)
        report["tokenizer"] = tokenizer_observation(provider._backend._tokenizer, cases["tokenizer_samples"])
        text = (REPO / "services/tests/fixtures/jina_passage_planning.txt").read_text(encoding="utf-8")
        context = {"provider_id": args.provider, "registry": str(args.registry.absolute()),
                   "description": report["description"], "manifest": report["manifest"],
                   "cases": cases, "planning_text": text}
        report["stage"] = "rust_transport"
        started = time.monotonic()
        report["rust"] = asyncio.run(rust_roundtrip(provider, context, output))
        report["rust_seconds"] = time.monotonic() - started
        # App shutdown unloads the provider. Require a new receipt and reject
        # work pinned to the former load; tokenizer recounts ran before unload.
        old = ProviderReceipt.from_dict(report["description"]["receipt"])
        current = provider.describe()
        require(current["receipt"]["space_id"] == old.space_id, "reload changed space")
        require(current["receipt"]["serving_generation"] != old.serving_generation, "reload reused generation")
        try:
            provider.embed_late(model=provider.model_name, task="retrieval.passage", text="old receipt",
                                boundaries=[[0, 3]], expected=old)
        except ReceiptMismatchError:
            report["stale_receipt_refused"] = True
        else:
            raise ValueError("stale receipt was accepted after reload")
        if args.arangod or args.docker:
            from qualify_verified_jina import database_roundtrip
            report["stage"] = "database"
            context.update(description=current, chunk_policy="late", suite_sha256=digest)
            server_args = ["--docker"] if args.docker else ["--arangod", str(args.arangod.absolute())]
            report["database"] = asyncio.run(database_roundtrip(provider, context, output, server_args))
        if args.device.startswith("cuda:"):
            report["runtime"]["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        report.update(passed=True, stage="complete")
    except (Exception, KeyboardInterrupt) as error:
        interrupted_run = isinstance(error, KeyboardInterrupt)
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2048]}
        print(f"Qualification failed during {report['stage']}: {error}", file=sys.stderr)
    finally:
        if provider is not None:
            try:
                provider.unload()
            except Exception as error:
                report.update(passed=False, cleanup_error=str(error)[:2048])
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(f"Qualification evidence: {output}", flush=True)
    return 130 if interrupted_run else (0 if report["passed"] else 1)


if __name__ == "__main__":
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("qualification interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
