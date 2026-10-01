"""CIFA feature-moment augmentation (Section III.B, Algorithm 1).

Implements the explicit content-baseline / style-deviation decomposition:

    instance moments  m(Z) = (mu, sigma)
    content baseline   b    : class-prototype (EMA) of moments
    style deviation    d    = m - b,   d independent of content C
    augmented moments  m~   = b + d_aug,   d_aug = lam*d + (1-lam)*d'
    recomposed feature T(Z) = sigma~ * (Z-mu)/sigma + mu~

Only the style deviation is resampled; the content baseline and the
normalized residual (Z-mu)/sigma are held fixed.
"""
import torch


def instance_moments(Z, eps):
    """Channel mean/std of feature tensor Z [B,C,H,W]."""
    mu = Z.mean(dim=(2, 3), keepdim=True)
    var = ((Z - mu) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma = torch.sqrt(var + eps)
    return mu, sigma


class BaselineTracker:
    """Per-class EMA prototypes of channel moments at each selected layer."""

    def __init__(self, num_classes, channels_by_layer, momentum=0.9, device="cpu"):
        self.momentum = momentum
        self.num_classes = num_classes
        self.layers = list(channels_by_layer.keys())
        self.proto_mu = {}
        self.proto_sigma = {}
        self.seen = {}
        for name, c in channels_by_layer.items():
            self.proto_mu[name] = torch.zeros(num_classes, c, device=device)
            self.proto_sigma[name] = torch.zeros(num_classes, c, device=device)
            self.seen[name] = torch.zeros(num_classes, device=device)
        # global fallback (EMA over every instance) for unseen classes
        self.global_mu = {name: torch.zeros(c, device=device)
                          for name, c in channels_by_layer.items()}
        self.global_sigma = {name: torch.zeros(c, device=device)
                             for name, c in channels_by_layer.items()}
        self.global_seen = {name: 0 for name in channels_by_layer}

    @torch.no_grad()
    def update(self, name, mu, sigma, y):
        """mu,sigma: [B,C] detached clean moments; y: [B]."""
        proto_mu = self.proto_mu[name]
        proto_sigma = self.proto_sigma[name]
        seen = self.seen[name]
        classes = torch.unique(y)
        for k in classes:
            idx = (y == k)
            bm = mu[idx].mean(dim=0)
            bs = sigma[idx].mean(dim=0)
            kk = int(k)
            if seen[kk] < 0.5:
                proto_mu[kk] = bm
                proto_sigma[kk] = bs
            else:
                m = self.momentum
                proto_mu[kk].mul_(m).add_(bm, alpha=1 - m)
                proto_sigma[kk].mul_(m).add_(bs, alpha=1 - m)
            seen[kk] = 1.0

        gm = mu.mean(dim=0)
        gs = sigma.mean(dim=0)
        if self.global_seen[name] == 0:
            self.global_mu[name] = gm
            self.global_sigma[name] = gs
        else:
            m = self.momentum
            self.global_mu[name].mul_(m).add_(gm, alpha=1 - m)
            self.global_sigma[name].mul_(m).add_(gs, alpha=1 - m)
        self.global_seen[name] = 1

    @torch.no_grad()
    def get_baseline(self, name, y):
        """Return b_mu,b_sigma [B,C], falling back to global for unseen."""
        b_mu = self.proto_mu[name][y].clone()
        b_sigma = self.proto_sigma[name][y].clone()
        unseen = (self.seen[name][y] < 0.5)
        if unseen.any():
            b_mu[unseen] = self.global_mu[name]
            b_sigma[unseen] = self.global_sigma[name]
        return b_mu, b_sigma


def _derangement(B, device):
    """Permutation with no fixed points (partner is always another instance)."""
    if B <= 1:
        return torch.randperm(B, device=device)
    while True:
        perm = torch.randperm(B, device=device)
        if (perm != torch.arange(B, device=device)).all():
            return perm


def _sample_deviation_gaussian(d):
    """Sample deviations from the batch's empirical Gaussian (DSU-style)."""
    mean = d.mean(dim=0, keepdim=True)
    std = d.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean + std * torch.randn_like(d)


def augment_layer(name, Z, y, tracker, cfg):
    """Apply the deviation-mixing operator at one layer.

    Z: [B,C,H,W] feature at the layer. Returns Z_aug [B,C,H,W].
    Statistics are treated as constants (computed under no_grad); gradient
    reaches the earlier layers only through the normalized residual
    (Z-mu)/sigma, exactly as in the paper.
    """
    eps = cfg["cifa_moments_eps"]
    sigma_min = cfg["cifa_sigma_min"]
    B = Z.shape[0]

    with torch.no_grad():
        mu, sigma = instance_moments(Z, eps)      # [B,C,1,1], constants
        mu_f = mu.flatten(1)
        sigma_f = sigma.flatten(1)

        if cfg["cifa_decompose"]:
            b_mu, b_sigma = tracker.get_baseline(name, y)  # [B,C]
            d_mu = mu_f - b_mu
            d_sigma = sigma_f - b_sigma
        else:
            # Ablation: mix the FULL moments (MixStyle/DSU-style). Equivalent
            # to a zero baseline with the deviation equal to the full moments,
            # so new_mu = lam*mu + (1-lam)*mu' (likewise for sigma).
            b_mu = torch.zeros_like(mu_f)
            b_sigma = torch.zeros_like(sigma_f)
            d_mu = mu_f
            d_sigma = sigma_f

        if cfg["cifa_mix_mode"] == "gaussian":
            da_mu = _sample_deviation_gaussian(d_mu)
            da_sigma = _sample_deviation_gaussian(d_sigma)
        else:
            perm = _derangement(B, Z.device)
            d_mu_p = d_mu[perm]
            d_sigma_p = d_sigma[perm]
            lo, hi = cfg["cifa_lambda_min"], cfg["cifa_lambda_max"]
            lam = torch.rand(B, 1, device=Z.device) * (hi - lo) + lo
            da_mu = lam * d_mu + (1.0 - lam) * d_mu_p
            da_sigma = lam * d_sigma + (1.0 - lam) * d_sigma_p

        new_mu = (b_mu + da_mu).unsqueeze(-1).unsqueeze(-1)
        new_sigma = (b_sigma + da_sigma).unsqueeze(-1).unsqueeze(-1)
        new_sigma = new_sigma.clamp_min(sigma_min)

    # outside no_grad: residual keeps the gradient path back to earlier layers
    Z_norm = (Z - mu) / sigma
    return new_sigma * Z_norm + new_mu
