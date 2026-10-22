#!/usr/bin/env python3
"""Qualify conventional Jina document/code workloads with approved offline local assets.

Uses a private Unix HTTP server and in-memory Rust profiles. Creates no database
and never changes installed services. Evidence requires a new output directory.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
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
    """Run production catalog/binding/transport code against the same observed load."""
    from embedding.code_qualification import recheck
    from qualification_transport import run_rust_contract
    return await run_rust_contract(
        provider, context, output, repo=REPO, target="verified_jina_workloads",
        test="registered_conventional_workloads", environment="HADES_VERIFIED_CODE", recheck=recheck,
    )


def main(argv=None):
    """Retain successful and failed evidence, including cleanup and implementation identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True, help="document serving ceiling")
    parser.add_argument("--code-max-input-tokens", type=int, required=True, help="separate code serving ceiling")
    args = parser.parse_args(argv)
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        parser.error("set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before starting")
    if args.device != "cpu" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("device must be cpu or cuda:<ordinal>")
    if not all(32 <= value <= 8192 for value in (args.max_input_tokens, args.code_max_input_tokens)):
        parser.error("qualification ceilings must be in 32..8192")
    if args.max_input_tokens == args.code_max_input_tokens:
        parser.error("qualification requires different workload ceilings to check independent admission")
    os.umask(0o077)
    output = args.output_dir.absolute()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = {"protocol": "hades-verified-code-qualification-v1", "passed": False,
              "started_at": datetime.now(timezone.utc).isoformat(), "stage": "preflight"}
    provider = None
    interrupted_run = False
    try:
        import torch
        from embedding.code_qualification import qualify_workloads
        from embedding.qualification import load_cases, require
        from embedding.registry import ModelRegistry
        from embedding.verified_code import VerifiedJinaCodeProvider
        from embedding.verified_jina import _versions
        if args.device.startswith("cuda:") and (
            not torch.cuda.is_available() or int(args.device[5:]) >= torch.cuda.device_count()
        ):
            raise RuntimeError("requested CUDA device is unavailable; CPU fallback is refused")
        report["implementation_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        report["implementation_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=REPO, text=True).strip())
        fixtures = REPO / "services/tests/fixtures"
        cases, digests = {}, {}
        for kind, name in (("documents", "jina_qualification.json"), ("code", "jina_code_qualification.json")):
            cases[kind], digests[kind] = load_cases(fixtures / name)
        report["suite_sha256"] = digests
        planning = {"documents": (fixtures / "jina_passage_planning.txt").read_text(encoding="utf-8"),
                    "code": "\n".join(row["text"] for row in cases["code"]["documents"]) * 4}
        report["planning_sha256"] = {kind: hashlib.sha256(text.encode()).hexdigest() for kind, text in planning.items()}
        registration = ModelRegistry(args.registry).load(args.provider)
        report["manifest"] = json.loads(registration.manifest.canonical_bytes())
        report["runtime"] = {"device": args.device, "libraries": _versions(), "cuda": torch.version.cuda}
        provider = VerifiedJinaCodeProvider(registration, device=args.device,
            use_fp16=args.device.startswith("cuda:"), max_input_tokens=args.max_input_tokens,
            code_max_input_tokens=args.code_max_input_tokens)
        require(provider._backend.device == args.device, "provider changed device")
        warnings.filterwarnings("error", message="Found missing adapter keys.*")
        if args.device.startswith("cuda:"):
            report["runtime"]["gpu"] = torch.cuda.get_device_name(args.device)
        report["stage"] = "provider"
        report["provider"] = {}
        description = qualify_workloads(provider, cases, report["provider"])
        context = {"provider_id": args.provider, "registry": str(args.registry.absolute()),
                   "description": description, "manifest": report["manifest"],
                   "cases": cases, "planning_texts": planning}
        report["stage"] = "rust_transport"
        started = time.monotonic()
        report["rust"] = asyncio.run(rust_roundtrip(provider, context, output))
        report["rust_seconds"] = time.monotonic() - started
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
