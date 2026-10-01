"""Segmented ResNet backbone for CIFA.

The ResNet is split into stem + layer1..layer4 so the feature-moment
operator can be inserted at style-dominant bottleneck outputs. A single
training step performs:
  - a clean path (its moments update the content baseline), and
  - an augmented path (reads the baseline, mixes only style deviations),
returning two sets of logits for soft-worst aggregation + consistency.
"""
import random as pyrandom

import torch.nn as nn
from torchvision import models

from cifa_aug import augment_layer, instance_moments

_BLOCK_LAYERS = ["layer1", "layer2", "layer3", "layer4"]


class CIFAResNet(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, num_classes=7,
                 layers=("layer1", "layer2"), freeze_bn=True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = getattr(models, backbone)(weights=weights)

        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool
        self.feat_dim = net.fc.in_features
        self.fc = nn.Linear(self.feat_dim, num_classes)

        self.aug_layers = list(layers)
        self._freeze_bn = freeze_bn

    def train(self, mode=True):
        super().train(mode)
        if mode and self._freeze_bn:
            # DomainBed keeps pretrained BatchNorm in eval mode
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
        return self

    def forward_path(self, x, y, tracker, augment, cfg):
        """Run one full path; collect clean moments or augment at L."""
        eps = cfg["cifa_moments_eps"]
        moments = {}
        x = self.stem(x)
        for name in _BLOCK_LAYERS:
            x = getattr(self, name)(x)
            if name in self.aug_layers:
                if augment:
                    x = augment_layer(name, x, y, tracker, cfg)
                else:
                    mu, sigma = instance_moments(x, eps)
                    moments[name] = (mu.flatten(1), sigma.flatten(1))
        x = self.avgpool(x).flatten(1)
        logits = self.fc(x)
        return logits, x, moments

    def forward(self, x, y, tracker, cfg):
        # clean path; update baseline with detached clean moments
        logits_c, feat_c, moments = self.forward_path(
            x, y, tracker, augment=False, cfg=cfg)
        if tracker is not None:
            for name, (mu, sigma) in moments.items():
                tracker.update(name, mu.detach(), sigma.detach(), y)

        logits_a = None
        if self.training and cfg["cifa_enabled"] and tracker is not None:
            # batch-level Bernoulli trigger
            if pyrandom.random() < cfg["cifa_prob"]:
                logits_a, _, _ = self.forward_path(
                    x, y, tracker, augment=True, cfg=cfg)
        return logits_c, logits_a
