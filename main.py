import gym
from gym import spaces

import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from environment import Base, Bomber, CombatMap,BomberBaseEnv,create_combat_map_from_file

import copy


class DQNWithResnet(nn.Module):
    # DQN with Resnet34 as feature extractor
    def __init__(self, h, w, outputs):
        super(DQNWithResnet, self).__init__()
        # 用卷积层+resnet提取全局的信息
        self.conv_enemy_base = nn.Conv2d(2, 1, kernel_size=3, stride=1)
        self.conv_friendly_base = nn.Conv2d(2, 1, kernel_size=3, stride=1)
        self.conv_bomber = nn.Conv2d(2, 1, kernel_size=3, stride=1)
        resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
        # 去掉最后一层全连接层
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling to get the correct size
        # 用全连接层提取局部的信息
        self.fc_nearby = nn.Linear(49 * 6, 2048)
        # 合并全局信息和局部信息
        self.fc_global = nn.Linear(512, 2048)
        self.fc = nn.Linear(2048+2048, outputs)

    def forward(self, enemy_base, friendly_base, bomber,bomber_coordinate):
        # enemy_base[0]: hp_map, enemy_base[1]: value_map
        # friendly_base[0]: ammo_map, friendly_base[1]: fuel_map
        # bomber[0]: ammo, bomber[1]: fuel
        # The input to resnet should be [batch_size, channels, height, width]

        enemy_global = F.relu(self.conv_enemy_base(enemy_base))
        friendly_global = F.relu(self.conv_friendly_base(friendly_base))
        bomber_global = F.relu(self.conv_bomber(bomber))

        def adjust_to_resnet_input(tensor, target_size=(224, 224)):
            _, h, w = tensor.shape
            target_h, target_w = target_size

            # Calculate padding
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)

            # Pad the tensor
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            padded_tensor = F.pad(tensor, padding, mode='constant', value=0)

            return padded_tensor

        enemy_global = adjust_to_resnet_input(enemy_global)
        friendly_global = adjust_to_resnet_input(friendly_global)
        bomber_global = adjust_to_resnet_input(bomber_global)

        total_global = torch.cat((enemy_global, friendly_global, bomber_global), dim=1).unsqueeze(0)
        global_features = self.resnet(total_global)
        global_features = self.avgpool(global_features)
        global_features = global_features.view(global_features.size(0), -1)  # Flatten

        # 获取bomber的坐标，提取周围7*7的信息，使用padding处理边界情况
        bomber_x, bomber_y = bomber_coordinate

        def get_padded_area(map, x, y, size=7):
            padding = size // 2
            padded_map = F.pad(map, (padding, padding, padding, padding), mode='constant', value=0)
            return padded_map[:, x:x + size, y:y + size].flatten()

        nearby_enemy_hp = get_padded_area(enemy_base[0], bomber_x, bomber_y)
        nearby_enemy_value = get_padded_area(enemy_base[1], bomber_x, bomber_y)
        nearby_friendly_ammo = get_padded_area(friendly_base[0], bomber_x, bomber_y)
        nearby_friendly_fuel = get_padded_area(friendly_base[1], bomber_x, bomber_y)
        nearby_bomber_ammo = get_padded_area(bomber[0], bomber_x, bomber_y)
        nearby_bomber_fuel = get_padded_area(bomber[1], bomber_x, bomber_y)

        nearby_info = torch.cat((
            nearby_enemy_hp,
            nearby_enemy_value,
            nearby_friendly_ammo,
            nearby_friendly_fuel,
            nearby_bomber_ammo,
            nearby_bomber_fuel), dim=0).unsqueeze(0)

        nearby_features = F.relu(self.fc_nearby(nearby_info))

        global_features = F.relu(self.fc_global(global_features))
        combined_features = torch.cat((global_features, nearby_features), dim=1)
        return self.fc(combined_features)


# Memory Replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# # debugging, check if map creation is correct
# test_map = create_combat_map_from_file("testcase/test2.txt")
# print(test_map.enemy_base)

