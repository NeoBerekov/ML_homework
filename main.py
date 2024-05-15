import gym
import gymnasium
from gymnasium import spaces
from gymnasium.envs.registration import register
import tianshou as ts

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
from environment import CombatMap,CombatEnv,create_combat_map_from_file

import copy

combatMap = create_combat_map_from_file("testcase/test2.txt")

register(
    id='Combat-v0',
    entry_point='environment:CombatEnv',
    kwargs={'combatMap':combatMap}
)

gym.make('Combat-v0',combatMap=combatMap)







