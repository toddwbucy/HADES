"""Versioned semantic identity for graph tensors and checkpoint parameters."""
from copy import deepcopy

FEATURE_POLICY = "node-or-codefile-mean-v1;missing=zero"
FIELDS = {"version", "relation_order", "collection_names", "feature_dim",
          "architecture", "feature_policy", "feature_models"}


def validate_contract(contract, model_config, feature_dim):
    if not isinstance(contract, dict) or set(contract) != FIELDS:
        raise ValueError("missing or malformed graph contract; rebuild graph/checkpoint with verified provenance")
    if type(contract["version"]) is not int or contract["version"] != 1:
        raise ValueError("unsupported graph contract version")
    if type(contract["feature_dim"]) is not int or contract["feature_dim"] <= 0 or contract["feature_dim"] != feature_dim:
        raise ValueError("graph contract feature dimension does not match tensors/model")
    if contract["architecture"] != (model_config.architecture or "rgcn"):
        raise ValueError("graph contract architecture does not match model")
    if contract["feature_policy"] != FEATURE_POLICY:
        raise ValueError("unsupported feature construction policy")
    for key, count in (("relation_order", model_config.num_relations),
                       ("collection_names", model_config.num_collection_types)):
        names = contract[key]
        if (not isinstance(names, list) or len(names) != count or not names
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError(f"invalid {key} mapping in graph contract")
    models = contract["feature_models"]
    if not isinstance(models, dict) or set(models) != set(contract["collection_names"]):
        raise ValueError("feature provenance must identify every collection")
    for identities in models.values():
        if (not isinstance(identities, list) or len(identities) > 1
                or any(not isinstance(name, str) or not name.strip() for name in identities)
                or identities != sorted(set(identities))):
            raise ValueError("feature model identities must be canonical nonempty strings")
    return deepcopy(contract)
