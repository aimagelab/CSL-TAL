import torch
import numpy as np
from typing import List, Tuple, Dict
from utils import unravel


class CSLTALLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.delay_align = args.delay_align
        self.detach = args.detach
        self.pseudo_label = args.pseudo_label
        self.num_samples = args.num_samples

        if self.pseudo_label:
            self.low_threshold = -1.0
            self.high_threshold = -1.0
            self.median = -1.0
            self.median_percentile = args.median_percentile

    def update_thresholds(self, attention: List[torch.Tensor], first_batch: bool) -> None:
        all_att = torch.cat(attention).squeeze().detach().cpu().numpy()
        if self.low_threshold == -1.0:
            self.low_threshold = np.percentile(all_att, 20)
            self.high_threshold = np.percentile(all_att, 80)
            self.median = np.percentile(all_att, self.median_percentile)
        else:
            self.low_threshold = 0.9 * self.low_threshold + 0.1 * np.percentile(all_att, 20)
            self.high_threshold = 0.9 * self.high_threshold + 0.1 * np.percentile(all_att, 80)
            self.median = 0.9 * self.median + 0.1 * np.percentile(all_att, self.median_percentile)

        if first_batch:
            self.current_low_threshold = self.low_threshold
            self.current_high_threshold = self.high_threshold
            self.current_median = self.median

    def forward(self, classification: torch.Tensor, attention: List[torch.Tensor],
                video_label: torch.Tensor, features: torch.Tensor,
                epoch: int, len_feat: Tuple,
                first_batch: bool = None) -> Tuple[torch.Tensor, Dict[str, float]]:

        device = video_label.device
        video_label_repeat = video_label

        align_loss = torch.tensor(0.0)
        if self.num_samples > 1 and len_feat:

            if self.pseudo_label:
                self.update_thresholds(attention, first_batch)

            video_label_repeat = video_label.repeat_interleave(self.num_samples)

            a = torch.cat(attention).split(unravel(len_feat))
            if epoch >= self.delay_align:
                # pairwise distance between num_samples attention
                if self.num_samples > 2:
                    each_sample_couple = [torch.mean(torch.nn.functional.pdist(torch.stack(b).squeeze(), p=2)**2) for b in [a[i:i + self.num_samples] for i in range(0, len(a), self.num_samples)]]
                elif self.detach and not self.pseudo_label:
                    each_sample_couple = [torch.mean((b[0] - b[1].detach())**2) for b in zip(a[0::2], a[1::2])]
                elif not self.detach and not self.pseudo_label:
                    each_sample_couple = [torch.mean((b[0] - b[1])**2) for b in zip(a[0::2], a[1::2])]
                elif self.pseudo_label:
                    each_sample_couple = []
                    for A, B in zip(a[0::2], a[1::2]):
                        mask = (A > self.current_high_threshold) | (A < self.current_low_threshold)
                        target = torch.zeros_like(A)
                        target[A >= self.current_median] = 1.

                        each_sample_couple.append(torch.nn.functional.binary_cross_entropy(B, target, reduction="none")[mask].mean())

                align_loss = self.gamma * (torch.stack(each_sample_couple).squeeze()).sum()
            else:
                align_loss = torch.tensor(0.0)

        smooth_loss = self.alpha * (torch.stack([(a[:-1] - a[1:]).pow(2).sum(dim=0) for a in attention])).sum()
        sparsity_loss = self.beta * (torch.stack([torch.linalg.norm(a, ord=1, dim=0) for a in attention]).squeeze()).sum()

        class_loss = torch.nn.functional.binary_cross_entropy(classification.squeeze().to(device), video_label_repeat.squeeze())

        loss = (align_loss + class_loss + smooth_loss + sparsity_loss) / len(features)
        losses = {"align_loss": align_loss.item(), "class_loss": class_loss.item(), "smooth_loss": smooth_loss.item(), "sparsity_loss": sparsity_loss.item()}
        return loss, losses
