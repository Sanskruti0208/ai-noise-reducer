import torch
import torch.nn as nn

class SimpleDenoisingCNN(nn.Module):
    def __init__(self):
        super(SimpleDenoisingCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=15, padding=7),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=15, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=15, padding=7)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
