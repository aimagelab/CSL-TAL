from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from model.loss import CSLTALLoss
from torch import Tensor
from utils import unravel, warn

from model.gated_attention import GatedAttention
from model.temporal_conv import MaskedConv1D


class STPN(nn.Module):
    def __init__(self, in_features: int = 2048, num_classes: int = 1, t_win: int = 5, transformer: bool = False, gated_att: bool = False):
        super(STPN, self).__init__()

        self.t_win = t_win
        self.transformer = transformer
        self.num_classes = num_classes

        if not gated_att:
            self.attention = nn.Sequential(
                Rearrange('seq f -> 1 f seq'),
                MaskedConv1D(in_features, in_features, kernel_size=5, stride=1, padding=2, bias=True),
                Rearrange('1 f seq -> seq f'),
                nn.Linear(in_features=in_features, out_features=1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=num_classes),
                nn.Sigmoid()
            )
        else:
            self.attention = nn.Sequential(
                Rearrange('seq f -> 1 f seq'),
                MaskedConv1D(in_features, in_features, kernel_size=5, stride=1, padding=2, bias=True),
                Rearrange('1 f seq -> seq f'),
                GatedAttention(in_features, num_classes)
            )

        if self.transformer:
            warn("Using transformer, you should first remove Conv1D from attention")
            encoder_layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=8)
            self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_classes),
            nn.Sigmoid()
        )

        self.init_weights()

    def init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, len_feat: Optional[Tuple[int]] = None) -> Tensor:

        if self.training:
            attentions = []
            len_feat = torch.tensor([0] + unravel(len_feat)).cumsum(0)
            scores = torch.zeros(len(len_feat) - 1, self.num_classes)

            for i in range(len(len_feat) - 1):
                features = x[len_feat[i]:len_feat[i + 1]]
                if self.transformer:
                    transformer_mask = torch.tril(torch.ones([len(features)] * 2)).to(features.device)
                    features = self.transformer_enc(features.unsqueeze(1), mask=transformer_mask).squeeze()

                attentions.append(self.attention(features))
                y = (attentions[i] * features).sum(dim=0)  # temporal aggregation
                scores[i] = self.classifier(y)

            return attentions, scores
        else:  # test
            if self.transformer:
                x = x.transpose(0, 1)
                transformer_mask = torch.tril(torch.ones([len(x)] * 2)).to(x.device)
                x = self.transformer_enc(x, mask=transformer_mask).transpose(0, 1)

            attention = self.attention(x.squeeze(0))
            x = (attention * x).sum(dim=1)  # temporal aggregation
            x = self.classifier(x)
            return attention, x

    def t_cam(self, x: Tensor) -> Tensor:
        """Used to compute the Temporal CAM of the single instance.

        Args:
            x (Tensor): instance features.

        Returns:
            Tensor: T_CAM of the instance.
        """
        return self.classifier(x)

    def get_loss_fn(self, args) -> CSLTALLoss:
        return CSLTALLoss(args)


if __name__ == '__main__':
    import torch
    a = torch.randn(78, 2048)
    model = STPN()
    model(a)
