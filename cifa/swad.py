"""SWAD: Domain Generalization by Seeking Flat Minima (Cha et al., NeurIPS 2021).

Two pieces:
  LossValley     : detects, from per-epoch hold-out loss, the start (loss
                   converged to a plateau) and end (loss rebounds beyond a
                   tolerance, or a max valley length) of a low-loss valley.
  AveragedModel  : dense running average of parameters over that valley.

Semantics follow the official SWAD implementation; no target-domain data
is used (only the in-domain hold-out split).
"""
from copy import deepcopy

import torch


class LossValley:
    def __init__(self, n_converge=3, n_tolerance=6,
                 tolerance_ratio=0.3, tolerance_epoch=None):
        self.n_converge = n_converge
        self.n_tolerance = n_tolerance
        self.tolerance_ratio = tolerance_ratio
        self.tolerance_epoch = tolerance_epoch
        self.reset()

    def reset(self):
        self.losses = []
        self.start = -1
        self.end = -1
        self.mean = None
        self.above = 0

    def _is_converged(self):
        if len(self.losses) < self.n_converge:
            return False
        window = self.losses[-self.n_converge:]
        # earliest loss in the window is no larger than any later one:
        # the loss has stopped decreasing -> plateau reached
        return window[0] <= min(window[1:])

    def _is_tolerance(self, idx, loss):
        if self.mean is None:
            segment = self.losses[self.start:]
            self.mean = sum(segment) / len(segment)
        if loss > self.mean * (1.0 + self.tolerance_ratio):
            self.above += 1
        else:
            self.above = 0
            self.mean = None
        if self.tolerance_epoch is not None and \
                idx - self.start >= self.tolerance_epoch:
            self.end = idx
        elif self.above >= self.n_tolerance:
            self.end = idx
        return self.end

    def update(self, idx, loss):
        self.losses.append(loss)
        if self.start < 0 and self._is_converged():
            self.start = idx - self.n_converge + 1
        if self.start >= 0 and self.end < 0:
            self._is_tolerance(idx, loss)
        return self.start, self.end

    def need_average(self, idx):
        return self.start >= 0 and (self.end < 0 or idx <= self.end)


class AveragedModel:
    """Incremental dense average of a model's trainable parameters."""

    def __init__(self, model):
        self.n = 0
        self.module = deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        avg_params = dict(self.module.named_parameters())
        for name, p in model.named_parameters():
            avg_params[name].add_((p.detach() - avg_params[name]) / self.n)
        # copy non-persistent / running buffers directly
        for name, b in model.named_buffers():
            self.module.get_buffer(name).copy_(b)

    def state_dict(self):
        return self.module.state_dict()
