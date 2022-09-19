import torch.nn as nn



class Learner(nn.Module):
    def __init__(self, layers, input_dim: int = 2048, output_dim: int = 1, drop_p: float = 0.0, activation: str = 'relu'):
        super(Learner, self).__init__()

        activation_fun: nn.Module = {"relu": nn.ReLU(inplace=True),
                                     "silu": nn.SiLU(inplace=True),
                                     "gelu": nn.GELU(),
                                     "leaky_relu": nn.LeakyReLU(inplace=True),
                                     "elu": nn.ELU(inplace=True)}[activation]

        if layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                activation_fun,
                nn.Dropout(drop_p),
                nn.Linear(512, output_dim),
                nn.Sigmoid(),
            )
        elif layers == 3:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 512),
                activation_fun,
                nn.Dropout(drop_p),
                nn.Linear(512, 32),
                activation_fun,
                nn.Dropout(drop_p),
                nn.Linear(32, output_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError("Only 2 or 3 layers are supported")

        self.weight_init()

    def weight_init(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.classifier(x)
