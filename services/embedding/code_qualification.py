"""Fixed conventional document/code qualification gates; no database or live service."""

import time

from .identity import ProviderReceipt
from .qualification import check_stored_plan, qualify, require, verify_adapters
from .verified_jina import ReceiptMismatchError
from .workload_contracts import WORKLOAD_CONTRACTS


def qualify_workloads(provider, cases, evidence, *, adapter_check=verify_adapters):
    """Qualify both selected adapters and refuse equal-width cross-workload receipts."""
    checked = {}
    reload_checks = []
    def check_once_per_load(current):
        """Share the checkpoint pass within one generation, never across reloads."""
        generation = current.describe()["workloads"]["documents"]["receipt"]["serving_generation"]
        if generation not in checked:
            checked[generation] = adapter_check(current)
        return checked[generation]

    for kind in WORKLOAD_CONTRACTS:
        evidence[kind] = {}
        qualify(provider, cases[kind], workload=kind, adapter_check=check_once_per_load,
                evidence=evidence[kind], reload_checks=reload_checks)
    provider.unload()
    started = time.monotonic()
    description = provider.describe()
    reload_seconds = time.monotonic() - started
    check_once_per_load(provider)
    evidence["adapter_checks"] = [{"serving_generation": generation, "tensors_verified": count}
                                  for generation, count in checked.items()]
    for check in reload_checks:
        check(description, reload_seconds)
    observations = description["workloads"]
    receipts = {kind: ProviderReceipt.from_dict(observations[kind]["receipt"])
                for kind in ("documents", "code")}
    documents, code = receipts["documents"], receipts["code"]
    require(documents.artifact_md5 == code.artifact_md5
            and documents.serving_generation == code.serving_generation
            and documents.space_id != code.space_id, "workloads do not share one load with distinct spaces")
    forwards = []
    hook = provider._backend._model.register_forward_pre_hook(lambda *_: forwards.append(True))
    try:
        for task, wrong in (("code", documents), ("retrieval.passage", code), ("retrieval.query", code)):
            try:
                provider.embed(model=provider.model_name, texts=["short input"], task=task, expected=wrong)
            except ReceiptMismatchError:
                pass
            else:
                raise ValueError("cross-workload receipt was accepted")
        require(not forwards, "cross-workload receipt reached inference")
    finally:
        hook.remove()
    evidence["cross_workload_receipts_rejected"] = 3
    evidence["description"] = description
    return description


def recheck(provider, context, result):
    """Recount Rust-accepted source spans before the private HTTP server unloads the model."""
    observed = provider.describe()["workloads"]
    return {kind: check_stored_plan(
        provider, context["planning_texts"][kind], result[kind]["planning"],
        task=contract.passage_task, expected=ProviderReceipt.from_dict(observed[kind]["receipt"]),
        prompt=contract.passage_prompt,
    ) for kind, contract in WORKLOAD_CONTRACTS.items()}
