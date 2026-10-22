"""Fixed conventional document/code qualification gates; no database or live service."""

from .identity import ProviderReceipt
from .qualification import check_stored_plan, qualify, require, verify_adapters
from .verified_code import CODE_PROMPT
from .verified_jina import ReceiptMismatchError


def qualify_workloads(provider, cases, evidence, *, adapter_check=verify_adapters):
    """Qualify both selected adapters and refuse equal-width cross-workload receipts."""
    for kind in ("documents", "code"):
        evidence[kind] = {}
        qualify(provider, cases[kind], workload=kind, adapter_check=adapter_check, evidence=evidence[kind])
    description = provider.describe()
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
    return {kind: check_stored_plan(provider, context["planning_texts"][kind],
                                    result[kind]["planning"],
                                    prompt=CODE_PROMPT if kind == "code" else None)
            for kind in ("documents", "code")}
