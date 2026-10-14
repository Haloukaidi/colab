"""Commit BN running-stat updates once per image across prompt chunks.

Call after_backward after each chunk's backward (never mutate saved BN tensors
between forward and backward). Create a new instance for every logical image.
Training-mode normalization and affine gradients remain enabled for all chunks.
"""
import torch


class BatchNormOncePerImage:
    def __init__(self, model):
        self.modules = [m for m in model.modules()
                        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                        and m.training and m.track_running_stats]
        self.saved = None

    @torch.no_grad()
    def after_backward(self):
        if self.saved is None:
            self.saved = [(m.running_mean.clone(), m.running_var.clone(),
                           m.num_batches_tracked.clone()) for m in self.modules]
        else:
            for module, values in zip(self.modules, self.saved):
                for name, value in zip(('running_mean', 'running_var', 'num_batches_tracked'), values):
                    getattr(module, name).copy_(value)
