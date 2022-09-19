from torch import Tensor
from typing import Optional
import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', alpha: float = -1, gamma: float = 2, num_classes: int = 2) -> None:
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)

        one_hot = torch.eye(self.num_classes)[target].to(target.device)
        p = torch.sigmoid(input)

        ce_loss = F.binary_cross_entropy(
            p, one_hot, reduction="none")
        p_t = p * one_hot + (1 - p) * (1 - one_hot)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', alpha: float = -1, gamma: float = 2, num_classes: int = 2) -> None:
        super(BinaryFocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.num_classes = num_classes

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)

        # if input.dim() == 2:
        #     target = target.unsqueeze(1)

        if not (torch.all(input > 0) and torch.all(input < 1)):
            input = torch.sigmoid(input)

        ce_loss = F.binary_cross_entropy(input, target.float(), reduction="none")

        p_t = input * target.float() + (1 - input) * (1 - target.float())
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


if __name__ == '__main__':
    pass
