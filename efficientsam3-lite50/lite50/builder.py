"""Experimental RV-M Lite50 overlay built on official EfficientSAM3."""
from dataclasses import dataclass
import torch
from torch import nn
from .modules import LiteProjectionHead, LiteFPNLevel, FactorizedTokenEmbedding

@dataclass(frozen=True)
class Lite50Config:
    text_embedding_rank: int = 64
    text_attention_layers: int = 2
    vision_fusion_layers: int = 4
    geometry_layers: int = 2
    projection_mid_channels: int = 256
    parameter_budget: int = 50_000_000

def num_params(module):
    return sum(p.numel() for p in module.parameters())

def _prune(layers, keep):
    if not 1 <= keep <= len(layers):
        raise ValueError((keep, len(layers)))
    if keep == 1:
        return nn.ModuleList([layers[-1]])
    indices = [round(i*(len(layers)-1)/(keep-1)) for i in range(keep)]
    assert len(indices) == len(set(indices))
    return nn.ModuleList([layers[i] for i in indices])

def make_lite50(model, config=Lite50Config(), initialize_text_from_dense=True):
    if model.inst_interactive_predictor is not None:
        raise ValueError("Optional SAM2 tracker not supported in Lite50")
    vision = model.backbone.vision_backbone
    student = vision.trunk.model
    if student.head[-1].out_channels != 1024:
        raise ValueError("Expected 1024-channel student features")
    c = student.head[0].in_channels
    student.head = LiteProjectionHead(c, config.projection_mid_channels, 1024)
    if len(vision.convs) != 4:
        raise ValueError("Expected four-level FPN")
    vision.convs = nn.ModuleList([
        LiteFPNLevel(scale=s) for s in (4.,2.,1.,.5)
    ])
    enc = model.backbone.language_backbone.encoder
    dense = enc.embedding_layer
    if not isinstance(dense, nn.Embedding):
        raise ValueError("Base MobileCLIP embedding expected")
    low = FactorizedTokenEmbedding(dense.num_embeddings, dense.embedding_dim,
                                   config.text_embedding_rank).to(
                                       device=dense.weight.device, dtype=dense.weight.dtype)
    if initialize_text_from_dense:
        low.initialize_from_dense(dense.weight)
    enc.embedding_layer = low
    # Official MobileCLIP-S0: RepMixer + four attention layers + RepMixer
    if len(enc.transformer) != 6:
        raise ValueError("Expected original MobileCLIP-S0 layout")
    layers = enc.transformer
    middle = _prune(nn.ModuleList(list(layers)[1:-1]), config.text_attention_layers)
    enc.transformer = nn.ModuleList([layers[0], *middle, layers[-1]])
    fusion = model.transformer.encoder
    fusion.layers = _prune(fusion.layers, config.vision_fusion_layers)
    fusion.num_layers = len(fusion.layers)
    for idx, layer in enumerate(fusion.layers):
        layer.layer_idx = idx
    geo = model.geometry_encoder
    geo.encode = _prune(geo.encode, config.geometry_layers)
    return model

def build_lite50_image_model(*, checkpoint_path=None, lite_checkpoint_path=None,
                             device="cpu", eval_mode=True, config=Lite50Config(),
                             strict_budget=True, initialize_text_from_dense=None,
                             **compat):
    from sam3.model_builder import build_efficientsam3_image_model
    fixed = dict(backbone_type="repvit", model_name="m1.1",
                 text_encoder_type="MobileCLIP-S0",
                 text_encoder_context_length=16,
                 enable_inst_interactivity=False, enable_segmentation=True)
    for key, val in fixed.items():
        if key in compat and compat[key] != val:
            raise ValueError(f"Lite50 requires {key}={val!r}")
    if initialize_text_from_dense is None:
        initialize_text_from_dense = checkpoint_path is not None
    model = build_efficientsam3_image_model(
        checkpoint_path=checkpoint_path, load_from_HF=False,
        device="cpu", eval_mode=eval_mode, **fixed)
    base_count = num_params(model)
    make_lite50(model, config, initialize_text_from_dense)
    count = num_params(model)
    model.lite50_original_params = base_count
    if strict_budget and count >= config.parameter_budget:
        raise RuntimeError(f"Parameter budget failed: {count:,} >= {config.parameter_budget:,}")
    if lite_checkpoint_path:
        data = torch.load(lite_checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(data.get("model", data), strict=True)
    model.to(device)
    model.eval() if eval_mode else model.train()
    return model

def save_lite50_checkpoint(model, path):
    torch.save(dict(model=model.state_dict(), variant="RV-M-Lite50",
                    params=num_params(model)), path)
