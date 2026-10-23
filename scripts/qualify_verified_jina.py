#!/usr/bin/env python3
"""Qualify an approved local Jina registration and a disposable Rust database flow.

Use the installed ML Python environment with HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1. No model download, live database discovery, or production
service activation is performed. The output directory must not already exist.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import warnings

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "services"))
sys.path.insert(0, str(REPO / "scripts"))

from embedding.qualification_run import (
    QualificationRun, install_interrupt_handler, require_device, validate_environment,
)


async def database_roundtrip(provider, context, output, server_args):
    """Share private provider ownership while the strict runner owns its disposable DB."""
    from qualification_transport import run_http_contract
    from embedding.identity import ProviderReceipt
    from embedding.qualification import check_stored_plan
    from embedding.workload_contracts import WORKLOAD_CONTRACTS

    def command(context_path, report_path):
        return ([sys.executable, str(REPO / "scripts/test_isolated_database.py"), *server_args,
                 "--verified-jina", str(context_path), "--verified-jina-report", str(report_path)], {})

    def recheck(current, ctx, result):
        if ctx.get("chunk_policy") == "late":
            from embedding.late_qualification import recheck as check_late
            checked = check_late(current, ctx["planning_text"], result["planning"])
        else:
            contract = WORKLOAD_CONTRACTS["documents"]
            checked = check_stored_plan(
                current, ctx["planning_text"], result["planning"]["plan"],
                task=contract.passage_task, prompt=contract.passage_prompt,
                expected=ProviderReceipt.from_dict(current.describe()["receipt"]),
            )
        result["planning"]["tokenizer_recheck"] = checked

    return await run_http_contract(
        provider, context, output, repo=REPO, name="database", label="isolated Rust round trip",
        command=command, recheck=recheck, timeout=1100, child_timeout=100,
    )


def main(argv=None):
    """Run fixed qualification gates and publish an explicit success or failure report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--device", required=True, help="explicit cpu or CUDA device; no fallback")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, required=True, help="new private evidence directory")
    server = parser.add_mutually_exclusive_group(required=True)
    server.add_argument("--arangod", type=Path, help="existing binary for a private Unix-only server")
    server.add_argument("--docker", action="store_true", help="use the runner's pinned isolated image")
    args = parser.parse_args(argv)
    validate_environment(parser, args)
    if not 32 <= args.max_input_tokens <= 8192:
        parser.error("qualification ceiling must be 32..8192 tokens")
    with QualificationRun(args.output_dir, REPO) as run:
        report, output = run.report, run.output
        import torch
        from embedding.qualification import load_cases, qualify
        from embedding.registry import ModelRegistry
        from embedding.verified_jina import VerifiedJinaProvider
        require_device(args.device)
        run.record_implementation()
        cases, digest = load_cases(REPO / "services/tests/fixtures/jina_qualification.json")
        report["suite_sha256"] = digest
        registration = ModelRegistry(args.registry).load(args.provider)
        report["manifest"] = json.loads(registration.manifest.canonical_bytes())
        provider = run.own(VerifiedJinaProvider(registration, device=args.device,
                                       use_fp16=args.device.startswith("cuda:"),
                                       max_input_tokens=args.max_input_tokens))
        if provider._backend.device != args.device:
            raise RuntimeError("provider selected a different device")
        warnings.filterwarnings("error", message="Found missing adapter keys.*")
        report["stage"] = "provider"
        print("Running fixed provider qualification gates...", flush=True)
        report["provider"] = {}
        qualify(provider, cases, evidence=report["provider"])
        report["stage"] = "database"
        print("Running verified HTTP -> Rust -> disposable database round trip...", flush=True)
        context = {"provider_id": args.provider, "registry": str(args.registry.absolute()),
                   "description": report["provider"]["description"],
                   "manifest": report["manifest"], "cases": cases, "suite_sha256": digest,
                   "planning_text": (REPO / "services/tests/fixtures/jina_passage_planning.txt").read_text(encoding="utf-8")}
        server_args = ["--docker"] if args.docker else ["--arangod", str(args.arangod.absolute())]
        report["database"] = asyncio.run(database_roundtrip(provider, context, output, server_args))
        report.update(passed=True, stage="complete")
    return run.exit_code


if __name__ == "__main__":
    install_interrupt_handler()
    raise SystemExit(main())
