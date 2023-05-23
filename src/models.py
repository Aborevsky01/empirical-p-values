import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class CNN_MNIST(nn.Module):
    def __init__(self, out_ch=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(8 * 7 * 7, out_ch)
        )
        
    def forward(self, x):
        return self.layers(x.reshape(-1, 28, 28).unsqueeze(1))
