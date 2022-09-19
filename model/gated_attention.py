import torch.nn as nn
import torch


class GatedAttention(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GatedAttention, self).__init__()
        self.L = in_features
        self.D = 256
        self.K = out_features

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.weights_init()

    def xavier_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.squeeze(0)

        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK

        return torch.sigmoid(A)


if __name__ == '__main__':
    x = torch.randn(100, 1024)
    a = GatedAttention(1024, 1)
    print(a(x))
