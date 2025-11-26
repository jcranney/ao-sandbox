from torch import nn
import os

WIDTH = int(os.environ.get("WIDTH", default="128"))
DEPTH = int(os.environ.get("DEPTH", default="2"))

class SimpleNet(nn.Module):
    width: int = WIDTH
    depth: int = DEPTH
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = [nn.Linear(16*16, self.width)]
        for _ in range(self.depth-1):
            layers += [nn.PReLU(), nn.Linear(self.width, self.width)]
        layers += [nn.PReLU(), nn.Linear(self.width, 10)]
        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
