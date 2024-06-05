import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, map_size,local_obs_window,map_depth, num_actions):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.map_size = map_size
        self.map_depth = map_depth
        self.local_obs_window = local_obs_window
        self.fc1 = nn.Linear(self.map_depth*self.local_obs_window*self.local_obs_window, 4096)
        self.fc2 = nn.Linear(4096,2048)
        self.fc3 = nn.Linear(2048, self.num_actions)

    def forward(self, local_obs):
        x = torch.flatten(local_obs, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# dummy_obs = torch.rand((1, 9, 6, 9))  # Batch size is 1
# model = DQN((6, 9), 9, 5)
# print(model(dummy_obs))

