import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPolicy(nn.Module):
    def __init__(
    		self,
    		observation_space_dim: int,
    		action_space_dim: int):
        super().__init__()
        self.l1 = nn.Linear(observation_space_dim, 8)
        self.dropout = nn.Dropout(p=0.2)
        self.l2 = nn.Linear(8, action_space_dim)

    def forward(self, batch):
        hidden = torch.tanh(self.dropout(self.l1(batch)))
        out = self.l2(hidden)
        return F.softmax(out, dim=-1)