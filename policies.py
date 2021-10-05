import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPolicy(nn.Module):
    def __init__(
            self,
            observation_space_dim: int,
            action_space_dim: int):
        super().__init__()
        self.l1 = nn.Linear(observation_space_dim, 256)
        self.dropout = nn.Dropout(p=0.2)
        self.l2 = nn.Linear(256, action_space_dim)

    def forward(self, batch):
        hidden = torch.relu(self.dropout(self.l1(batch)))
        out = self.l2(hidden)
        return F.softmax(out, dim=-1)


class LinearPolicy(nn.Module):
    def __init__(
            self,
            observation_space_dim: int,
            action_space_dim: int):
        super().__init__()
        self.l1 = nn.Linear(observation_space_dim, action_space_dim, bias=False)

    def forward(self, batch):
        out = self.l1(batch)
        return F.softmax(out, dim=-1)

class ConvPolicy(nn.Module):
    def __init__(
            self,
            observation_space_dim: int,
            action_space_dim: int):
        super().__init__()


        self.features = nn.Sequential(
            nn.Conv2d(observation_space_dim, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, action_space_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.softmax(x, dim=-1)

