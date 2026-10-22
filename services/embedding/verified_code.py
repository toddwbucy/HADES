"""Opt-in conventional document/code observations for one registered Jina load."""

import copy

from .identity import embedding_space_id
from .jina_v4 import MAX_TOKENS
from .verified_jina import BehaviorChangedError, VerifiedJinaProvider
from .workload_contracts import WORKLOAD_CONTRACTS

CODE_CONTRACT = WORKLOAD_CONTRACTS["code"]
CODE_PROMPT = CODE_CONTRACT.passage_prompt


class VerifiedJinaCodeProvider(VerifiedJinaProvider):
    """Serve two separately receipted conventional spaces, never implicit late code.

    Construction is operator-owned and requires an explicit code budget. The
    existing document-only and contextual factories retain their wire contracts.
    """

    embedding_tasks = (*VerifiedJinaProvider.embedding_tasks, *CODE_CONTRACT.prompts)
    passage_tasks = (*VerifiedJinaProvider.passage_tasks, CODE_CONTRACT.passage_task)

    def __init__(self, registration, *, code_max_input_tokens, **kwargs):
        """Validate the code budget independently before loading any model artifacts."""
        if (type(code_max_input_tokens) is not int
                or not 1 <= code_max_input_tokens <= min(MAX_TOKENS, 32768)):
            raise ValueError("code token budget must fit the configured Jina serving ceiling")
        super().__init__(registration, **kwargs)
        self._code_max_input_tokens = code_max_input_tokens

    def _derive(self):
        """Observe both actual adapters under the same artifact and serving generation."""
        document = super()._derive()
        model = self._backend._model
        if CODE_CONTRACT.adapter not in model.config.task_names:
            raise BehaviorChangedError("loaded model does not implement the Jina code contract")
        if self._code_max_input_tokens > min(
            model.config.max_position_embeddings, self._backend._tokenizer.model_max_length
        ):
            raise BehaviorChangedError("code serving budget exceeds loaded model/tokenizer context")
        identity = document.pop("model")
        code = copy.deepcopy(document)
        for role in ("passage", "query"):
            code["space"][role] = CODE_CONTRACT.behavior(role)
        code["receipt"]["space_id"] = embedding_space_id(code["space"])
        code["capabilities"].update(tasks=list(CODE_CONTRACT.prompts),
                                    specialized_code=CODE_CONTRACT.specialized_code,
                                    max_input_tokens=self._code_max_input_tokens)
        return {"format": "hades-provider-description-v2", "model": identity,
                "workloads": {"documents": document, "code": code}}

    def _workload(self, description, task):
        """Select the explicit observation; never borrow the other workload's receipt."""
        if task not in self.embedding_tasks:
            raise ValueError("unsupported verified workload task")
        kind = next(kind for kind, contract in WORKLOAD_CONTRACTS.items() if task in contract.prompts)
        return description["workloads"][kind]
