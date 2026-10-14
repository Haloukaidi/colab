"""Expensive optional smoke-test audit; never enabled implicitly in long training."""
import hashlib
import torch
from .train_policy import trainable_modules


def fingerprints(model):
    return {n: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, v in model.state_dict().items()}


def frozen_keys(model):
    active = trainable_modules(model, model._lite50_phase)
    allowed = {id(x) for module in active.values() for x in list(module.parameters()) + list(module.buffers())}
    return {n for n, x in list(model.named_parameters()) + list(model.named_buffers()) if id(x) not in allowed}


def verify_update(model, before, frozen):
    after = fingerprints(model)
    changed = {k for k in before if before[k] != after[k]}
    if changed & frozen:
        raise AssertionError(f"Frozen state changed: {sorted(changed & frozen)[:10]}")
    names = {id(p): n for n, p in model.named_parameters()}
    groups = {}
    active = trainable_modules(model, model._lite50_phase)
    if model._lite50_phase != "vision":
        embedding = model.backbone.language_backbone.encoder.embedding_layer
        active = {**active, "low_rank_codes": embedding.codes,
                  "low_rank_projection": embedding.projection}
    for label, module in active.items():
        params = list(module.parameters())
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
            raise AssertionError(f"Non-finite gradient in {label}")
        nonzero = sum(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in params)
        updates = sum(names[id(p)] in changed for p in params)
        if not nonzero or not updates:
            raise AssertionError(f"No effective update for {label}: gradients={nonzero}, updates={updates}")
        groups[label] = {"nonzero_gradient_tensors": nonzero, "updated_tensors": updates}
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise AssertionError("Frozen parameter has gradient")
    return {"phase": model._lite50_phase, "groups": groups,
            "frozen_tensors_checked": len(frozen & before.keys()), "frozen_unchanged": True}
