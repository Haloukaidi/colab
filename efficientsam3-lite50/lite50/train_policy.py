"""Explicit Lite50 curriculum. Frozen weights AND buffers remain unchanged."""
import types
import torch

PHASES = ("vision", "text", "geometry", "fusion1", "fusion2", "fusion4")


def trainable_modules(model, phase):
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    modules = {}
    if phase != "text":
        modules["projection"] = model.backbone.vision_backbone.trunk.model.head
        neck = model.backbone.vision_backbone
        count = getattr(neck, "active_levels", None)
        if count is None:
            count = len(neck.convs) - model.backbone.scalp
        for i in range(count):
            modules[f"fpn{i}"] = neck.convs[i]
    if phase not in ("vision", "geometry"):
        modules["text"] = model.backbone.language_backbone
    if phase.startswith("fusion"):
        count = int(phase[-1])
        encoder = model.transformer.encoder
        if len(encoder.layers) != 4:
            raise ValueError("Expected the four-layer Lite50 fusion encoder")
        # Parameters outside layers (if present) only enter the full-fusion stage.
        if count == 4:
            modules["fusion"] = encoder
        else:
            for i in range(4-count, 4):
                modules[f"fusion{i}"] = encoder.layers[i]
    return modules


def apply_train_policy(model, phase):
    groups = trainable_modules(model, phase)
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    for module in groups.values():
        module.requires_grad_(True)
    model._lite50_phase = phase
    if not hasattr(model, "_lite50_original_train"):
        model._lite50_original_train = model.train

    def train(self, mode=True):
        self._lite50_original_train(False)
        self.training = bool(mode)  # SAM3 uses this to perform GT matching.
        if mode:
            for module in trainable_modules(self, self._lite50_phase).values():
                module.train(True)
            if self._lite50_phase.startswith("fusion"):
                # Enable the upstream encoder's activation checkpoint wrapper,
                # without recursively enabling dropout in still-frozen layers.
                self.transformer.encoder.training = True
        return self

    model.train = types.MethodType(train, model)
    model.train(True)
    return parameter_report(model)


def parameter_report(model):
    return {"phase": model._lite50_phase,
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "groups": {n: sum(p.numel() for p in m.parameters() if p.requires_grad)
                       for n, m in trainable_modules(model, model._lite50_phase).items()}}


def assert_optimizer_coverage(model, optimizer):
    params = [p for g in optimizer.param_groups for p in g["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = {id(p) for p in params}
    if len(params) != len(actual) or actual != expected:
        raise RuntimeError(f"Optimizer coverage mismatch: missing={len(expected-actual)}, "
                           f"extra={len(actual-expected)}, duplicates={len(params)-len(actual)}")
    if not expected:
        raise RuntimeError("No trainable parameters")


def build_optimizer(model, lr=1e-4, weight_decay=0.01):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)
    assert_optimizer_coverage(model, optimizer)
    return optimizer
