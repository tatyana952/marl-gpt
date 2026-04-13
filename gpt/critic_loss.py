# Article: Farebrother, J., Orbay, J., Vuong, Q., Taïga, A.A., Chebotar, Y., Xiao, T., Irpan, A., Levine, S., Castro, P.S., Faust, A. and Kumar, A., 2024. Stop regressing: Training value functions via classification for scalable deep rl. arXiv preprint arXiv:2403.03950.
# https://arxiv.org/abs/2403.03950

import torch
import torch.special
import torch.nn as nn
import torch.nn.functional as F


class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float32
        )

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, mask=None, vec_loss=False
    ) -> torch.Tensor:
        # TODO: bins
        new_logits = logits
        new_target = self.transform_to_probs(target)
        if mask is not None:
            new_logits = logits[mask]
            new_target = new_target[mask]
        new_logits = new_logits.reshape(-1, self.num_bins)
        if vec_loss:
            return F.cross_entropy(new_logits, new_target, reduction="none")
        else:
            return F.cross_entropy(new_logits, new_target)

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        support = self.support.to(target.device)
        cdf_evals = torch.special.erf(
            (support - target.unsqueeze(-1))
            / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        support = self.support.to(probs.device)
        centers = (support[:-1] + support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)


class BinLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.support = torch.linspace(
            min_value, max_value, num_bins + 1, dtype=torch.float32
        )

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, mask=None, vec_loss=False
    ) -> torch.Tensor:
        new_logits = logits
        new_target = self.transform_to_probs(target)
        if logits.shape[-1] > self.num_bins:
            new_logits = logits[..., : self.num_bins]
        if mask is not None:
            new_logits = new_logits[mask]
            new_target = new_target[mask]
        new_logits = new_logits.reshape(-1, self.num_bins)
        if vec_loss:
            return F.cross_entropy(new_logits, new_target, reduction="none")
        else:
            return F.cross_entropy(new_logits, new_target)

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        support = self.support.unsqueeze(0)
        new_target = target.unsqueeze(-1)
        bin_probs = (support > new_target)[..., 1:] & (support <= new_target)[..., :-1]
        return bin_probs.to(torch.float)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)

    def target_is_pred(self, pred: torch.Tensor, target: torch.Tensor):
        pred_probs = self.transform_to_probs(pred)
        target_prob = self.transform_to_probs(target)
        return (pred_probs == target_prob).all(dim=-1)
