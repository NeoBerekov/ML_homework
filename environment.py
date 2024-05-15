import gymnasium as gym
from gymnasium import spaces

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

import copy

class Base:
    def __init__(self, x, y, ammo, fuel, hp, value, is_friendly):
        self.x = x
        self.y = y
        self.ammo = ammo
        self.fuel = fuel
        self.hp = hp
        self.value = value
        self.is_friendly = is_friendly


class Bomber:
    def __init__(self, ID, x, y, max_ammo, max_fuel, ammo=0, fuel=0,is_moved=0):
        self.ID = ID
        self.x = x
        self.y = y
        self.max_ammo = max_ammo
        self.max_fuel = max_fuel
        self.ammo = ammo
        self.fuel = fuel
        self.is_moved = is_moved
        self.reward = 0


class CombatMap:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.dict_friendly_base = {}
        self.dict_enemy_base = {}
        self.dict_bomber = {}
        self.ammo_map = np.zeros((self.width, self.height))
        self.fuel_map = np.zeros((self.width, self.height))
        self.hp_map = np.zeros((self.width, self.height))
        self.value_map = np.zeros((self.width, self.height))
        self.bomber_ammo_map = np.zeros((self.width, self.height))
        self.bomber_fuel_map = np.zeros((self.width, self.height))
        self.bomber_is_moved_map = np.zeros((self.width, self.height))

    def add_base(self, base):
        if base.is_friendly:
            self.dict_friendly_base[(base.x, base.y)] = base
        else:
            self.dict_enemy_base[(base.x, base.y)] = base

    def add_bomber(self, ID, x, y, max_ammo, max_fuel):
        self.dict_bomber[ID] = Bomber(ID, x, y, max_ammo, max_fuel)

    def add_friendly_base(self, x, y, ammo, fuel, hp, value):
        self.dict_friendly_base[(x, y)] = Base(x, y, ammo, fuel, hp, value, True)

    def add_enemy_base(self, x, y, ammo, fuel, hp, value):
        self.dict_enemy_base[(x, y)] = Base(x, y, ammo, fuel, hp, value, False)

    def create_ammo_map(self):
        for key in self.dict_friendly_base:
            self.ammo_map[key[0]][key[1]] = self.dict_friendly_base[key].ammo

    def create_fuel_map(self):
        for key in self.dict_friendly_base:
            self.fuel_map[key[0]][key[1]] = self.dict_friendly_base[key].fuel

    def create_hp_map(self):
        for key in self.dict_enemy_base:
            self.hp_map[key[0]][key[1]] = self.dict_enemy_base[key].hp

    def create_value_map(self):
        for key in self.dict_enemy_base:
            self.value_map[key[0]][key[1]] = self.dict_enemy_base[key].value

    def create_bomber_map(self):
        for key in self.dict_bomber:
            self.bomber_ammo_map[key[0]][key[1]] = self.dict_bomber[key].ammo
            self.bomber_fuel_map[key[0]][key[1]] = self.dict_bomber[key].fuel
            self.bomber_is_moved_map[key[0]][key[1]] = self.dict_bomber[key].is_moved

    def init_env(self, baseList):
        for base in baseList:
            self.add_base(base)


def create_combat_map_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    width, height = map(int, lines[0].strip().split())
    combat_map = CombatMap(width, height)

    current_line = 1

    # 读取蓝方基地
    num_friendly_bases = int(lines[current_line].strip())
    current_line += 1

    for _ in range(num_friendly_bases):
        x, y = map(int, lines[current_line].strip().split())
        current_line += 1
        ammo, fuel, hp, value = map(int, lines[current_line].strip().split())
        combat_map.add_friendly_base(x, y, ammo, fuel, hp, value)
        current_line += 1

    # 读取红方基地
    num_enemy_bases = int(lines[current_line].strip())
    current_line += 1

    for _ in range(num_enemy_bases):
        x, y = map(int, lines[current_line].strip().split())
        current_line += 1
        ammo, fuel, hp, value = map(int, lines[current_line].strip().split())
        combat_map.add_enemy_base(x, y, ammo, fuel, hp, value)
        current_line += 1

    # 读取战斗机信息
    num_bombers = int(lines[current_line].strip())
    current_line += 1

    for i in range(num_bombers):
        x, y, max_fuel, max_ammo = map(int, lines[current_line].strip().split())
        combat_map.add_bomber(i, x, y, max_fuel, max_ammo)
        current_line += 1

    # 创建地图信息
    combat_map.create_ammo_map()
    combat_map.create_fuel_map()
    combat_map.create_hp_map()
    combat_map.create_value_map()
    combat_map.create_bomber_map()

    return combat_map


class CombatEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, combat_map):
        super(CombatEnv, self).__init__()
        self.hp_map = combat_map.hp_map
        self.value_map = combat_map.value_map * 10 # increase the value to make the reward more significant
        self.ammo_map = combat_map.ammo_map
        self.fuel_map = combat_map.fuel_map
        self.bomber_ammo_map = combat_map.bomber_ammo_map
        self.bomber_fuel_map = combat_map.bomber_fuel_map
        self.bomber_is_moved_map = combat_map.bomber_is_moved_map
        self.agents = list(combat_map.dict_bomber.values())
        self.n_agents = len(self.agents)
        self.action_space = spaces.Discrete(9)
        self.original_combat_map = copy.deepcopy(combat_map)
        self.turn = 0
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(combat_map.width, combat_map.height, 7),
                                            dtype=np.uint16)

    def reset(self):
        combat_map = copy.deepcopy(self.original_combat_map)
        self.hp_map = combat_map.hp_map
        self.value_map = combat_map.value_map * 10
        self.ammo_map = combat_map.ammo_map
        self.fuel_map = combat_map.fuel_map
        self.bomber_ammo_map = combat_map.bomber_ammo_map
        self.bomber_fuel_map = combat_map.bomber_fuel_map
        self.bomber_is_moved_map = combat_map.bomber_is_moved_map
        self.agents = list(combat_map.dict_bomber.values())
        self.n_agents = len(self.agents)
        self.turn = 0
        return self._get_obs()

    def render(self, mode='human'):
        grid = np.full((self.original_combat_map.width, self.original_combat_map.height), '·'
                       , dtype='<U1')

        # Mark bases
        for (x, y), base in self.original_combat_map.dict_friendly_base.items():
            grid[x, y] = '*'
        for (x, y), base in self.original_combat_map.dict_enemy_base.items():
            grid[x, y] = '#'

        # Mark bombers
        for agent in self.agents:
            grid[agent.x, agent.y] = 'Δ'

        for row in grid:
            print(' '.join(row))
        print()

    def _move(self,agent_idx,direction): # return if the move is valid
        agent = self.agents[agent_idx]
        if agent.is_moved == 1:
            return False
        if direction == 0:  # move up
            new_pos = [agent.x, agent.y + 1]
        elif direction == 1:  # move down
            new_pos = [agent.x, agent.y - 1]
        elif direction == 2:  # move left
            new_pos = [agent.x - 1, agent.y]
        else:  # move right
            new_pos = [agent.x + 1, agent.y]

        if (new_pos[0] < 0
                or new_pos[0] >= self.original_combat_map.width
                or new_pos[1] < 0
                or new_pos[1] >= self.original_combat_map.height):
            return False

        if (self.value_map[new_pos[0]][new_pos[1]] != 0
                or self.bomber_ammo_map[new_pos[0]][new_pos[1]] != 0):
            # which means there is an enemy base or another bomber
            return False
        self.bomber_is_moved_map[agent.x][agent.y] = 0
        self.bomber_ammo_map[new_pos[0]][new_pos[1]] = self.bomber_ammo_map[agent.x][agent.y]
        self.bomber_fuel_map[new_pos[0]][new_pos[1]] = self.bomber_fuel_map[agent.x][agent.y]
        self.bomber_ammo_map[agent.x][agent.y] = 0
        self.bomber_fuel_map[agent.x][agent.y] = 0
        agent.x = new_pos[0]
        agent.y = new_pos[1]
        agent.fuel -= 1
        agent.is_moved = 1
        self.bomber_is_moved_map[agent.x][agent.y] = 1
        return True

    def _fire(self,agent_idx,direction): # return if the fire is valid and the reward
        agent = self.agents[agent_idx]
        if agent.ammo == 0:
            return False , -10
        x = agent.x
        y = agent.y
        if direction == 0:  # fire up
            new_pos = [x, y + 1]
        elif direction == 1:  # fire down
            new_pos = [x, y - 1]
        elif direction == 2:  # fire left
            new_pos = [x - 1, y]
        else:  # fire right
            new_pos = [x + 1, y]

        if (new_pos[0] < 0
                or new_pos[0] >= self.original_combat_map.width
                or new_pos[1] < 0
                or new_pos[1] >= self.original_combat_map.height):
            return False , -10

        if self.value_map[new_pos[0]][new_pos[1]] == 0:
            return False , -10

        self.hp_map[new_pos[0]][new_pos[1]] -= 1
        if self.hp_map[new_pos[0]][new_pos[1]] == 0:
            reward = self.value_map[new_pos[0]][new_pos[1]]
            self.value_map[new_pos[0]][new_pos[1]] = 0
            return True , reward
        else:
            return True , 1

    def _supply(self,agent_idx): # return if the supply is valid and reward
        agent = self.agents[agent_idx]
        x = agent.x
        y = agent.y
        reward = 0
        ammo_need = agent.max_ammo - agent.ammo
        fuel_need = agent.max_fuel - agent.fuel
        if ammo_need == 0 and fuel_need == 0:
            return False , -10
        if (x, y) in self.original_combat_map.dict_friendly_base:
            if ammo_need > 0:
                ammo_supply = self.ammo_map[x][y]
                if ammo_supply >= ammo_need:
                    agent.ammo = agent.max_ammo
                    self.ammo_map[x][y] -= ammo_need
                    reward += ammo_need
                else:
                    agent.ammo += ammo_supply
                    self.ammo_map[x][y] = 0
                    reward += ammo_supply
            if fuel_need > 0:
                fuel_supply = self.fuel_map[x][y]
                if fuel_supply >= fuel_need:
                    agent.fuel = agent.max_fuel
                    self.fuel_map[x][y] -= fuel_need
                    reward += fuel_need
                else:
                    agent.fuel += fuel_supply
                    self.fuel_map[x][y] = 0
                    reward += fuel_supply
            return True,reward
        else:
            return False,-10


    def step(self,actions):
        # actions is an array of actions for each agent
        assert len(actions) == self.n_agents
        rewards = np.zeros(self.n_agents)
        done = False
        for i in range(self.n_agents):
            if actions[i] == 0:
                if self._move(i,0):
                    rewards[i] += 0
            elif actions[i] == 1:
                if self._move(i,1):
                    rewards[i] += 0
            elif actions[i] == 2:
                if self._move(i,2):
                    rewards[i] += 0
            elif actions[i] == 3:
                if self._move(i,3):
                    rewards[i] += 0
            elif actions[i] == 4:
                valid , reward = self._fire(i,0)
                rewards[i] += reward
            elif actions[i] == 5:
                valid , reward = self._fire(i,1)
                rewards[i] += reward
            elif actions[i] == 6:
                valid , reward = self._fire(i,2)
                rewards[i] += reward
            elif actions[i] == 7:
                valid , reward = self._fire(i,3)
                rewards[i] += reward
            elif actions[i] == 8:
                valid , reward = self._supply(i)
                rewards[i] += reward
            else:
                rewards[i] += -100000

        if all(agent.is_moved == 1 for agent in self.agents):
            # 在回合结束时，检查done情况
            for agent in self.agents:
                agent.is_moved = 0
            self.turn += 1
            done = self._check_done()

        observation = self._get_obs()

        unmoved_agents = [agent for agent in self.agents if agent.is_moved == 0]
        # Additional info
        info = {"turn": self.turn,"actions":actions,"unmoved_agents":unmoved_agents}

        return observation, rewards, done, info

    def _get_obs(self):
        obs = np.zeros((self.original_combat_map.width, self.original_combat_map.height, 7), dtype=np.uint16)
        for agent in self.agents:
            obs[agent.x, agent.y, 0] = agent.ammo
            obs[agent.x, agent.y, 1] = agent.fuel
            obs[agent.x, agent.y, 2] = agent.is_moved
        obs[:, :, 3] = self.hp_map
        obs[:, :, 4] = self.value_map
        obs[:, :, 5] = self.ammo_map
        obs[:, :, 6] = self.fuel_map
        return obs

    def _check_done(self):
        # 所有敌人基地被摧毁，或者所有飞机都没油了
        return np.all(self.hp_map == 0) or np.all(self.bomber_fuel_map == 0)

    def close(self):
        pass