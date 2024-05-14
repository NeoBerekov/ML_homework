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
    def __init__(self, ID, x, y, max_ammo, max_fuel, ammo=0, fuel=0,is_moved=False):
        self.ID = ID
        self.x = x
        self.y = y
        self.max_ammo = max_ammo
        self.max_fuel = max_fuel
        self.ammo = ammo
        self.fuel = fuel
        self.is_moved = is_moved


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
        self.is_passable_map = np.ones((self.width, self.height))

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

    def create_is_passable_map(self):
        for key in self.dict_enemy_base:
            self.is_passable_map[key[0]][key[1]] = 0



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
    combat_map.create_is_passable_map()

    return combat_map
class BomberBaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,combat_map):
        super(BomberBaseEnv, self).__init__()
        self.action_space = spaces.Discrete(9)
        self.combat_map = combat_map
        self.bomber = list(self.combat_map.dict_bomber.values())[0]
        self.original_combat_map = copy.deepcopy(combat_map)
        self.turn = 0
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.combat_map.width, self.combat_map.height, 4),
                                            dtype=np.uint8)

    def step(self, action):
        # Execute one time step within the environment
        # action: 0-8, 0-8对应9个动作
        # 返回observation, reward, done, info
        # 0-3: 上下左右
        # 移动会导致fuel减少，并结束回合
        # 4-7: 上下左右开火
        # 射程一格，如果击中敌方基地，敌方基地hp减少，敌方基地hp为0时，敌方基地消失并奖励对应的value；开火会减少ammo，不会结束回合
        # 8: 在基地补给
        reward = 0
        info = {}
        flag_moved = False
        if action < 4:
            # 移动
            if action == 0:
                if (self.bomber.y > 0
                        and self.combat_map.is_passable_map[self.bomber.x][self.bomber.y-1] == 1):
                    self.bomber.y -= 1
                    self.bomber.fuel -= 1
                    self.turn += 1
                    flag_moved = True
                else:
                    reward = -10000
            elif action == 1:
                if (self.bomber.y < self.combat_map.height - 1
                        and self.combat_map.is_passable_map[self.bomber.x][self.bomber.y+1] == 1):
                    self.bomber.y += 1
                    self.bomber.fuel -= 1
                    self.turn += 1
                    flag_moved = True
                else:
                    reward = -10000
            elif action == 2:
                if (self.bomber.x > 0
                        and self.combat_map.is_passable_map[self.bomber.x-1][self.bomber.y] == 1):
                    self.bomber.x -= 1
                    self.bomber.fuel -= 1
                    self.turn += 1
                    flag_moved = True
                else:
                    reward = -10000
            elif action == 3:
                if (self.bomber.x < self.combat_map.width - 1
                        and self.combat_map.is_passable_map[self.bomber.x+1][self.bomber.y] == 1):
                    self.bomber.x += 1
                    self.bomber.fuel -= 1
                    self.turn += 1
                    flag_moved = True
                else:
                    reward = -10000
        elif action < 8:
            # 开火
            if self.bomber.ammo<=0:
                reward = -10000
            else:
                if action == 4:
                    if self.bomber.y > 0:
                        if self.combat_map.value_map[self.bomber.x][self.bomber.y - 1] != 0:
                            self.combat_map.hp_map[self.bomber.x][self.bomber.y - 1] -= 1
                            if self.combat_map.hp_map[self.bomber.x][self.bomber.y - 1] == 0:
                                reward += self.combat_map.value_map[self.bomber.x][self.bomber.y - 1]
                                self.combat_map.value_map[self.bomber.x][self.bomber.y - 1] = 0
                                self.combat_map.is_passable_map[self.bomber.x][self.bomber.y - 1] = 1
                                del self.combat_map.dict_enemy_base[(self.bomber.x, self.bomber.y - 1)]
                            self.bomber.ammo -= 1
                            reward += 1
                        else:
                            reward = -10000
                    else:
                        reward = -10000
                elif action == 5:
                    if self.bomber.y < self.combat_map.height - 1:
                        if self.combat_map.value_map[self.bomber.x][self.bomber.y + 1] != 0:
                            self.combat_map.hp_map[self.bomber.x][self.bomber.y + 1] -= 1
                            if self.combat_map.hp_map[self.bomber.x][self.bomber.y + 1] == 0:
                                reward += self.combat_map.value_map[self.bomber.x][self.bomber.y + 1]
                                self.combat_map.value_map[self.bomber.x][self.bomber.y + 1] = 0
                                self.combat_map.is_passable_map[self.bomber.x][self.bomber.y + 1] = 1
                                del self.combat_map.dict_enemy_base[(self.bomber.x, self.bomber.y + 1)]
                            self.bomber.ammo -= 1
                            reward += 1
                        else:
                            reward = -10000
                    else:
                        reward = -10000
                elif action == 6:
                    if self.bomber.x > 0:
                        if self.combat_map.value_map[self.bomber.x - 1][self.bomber.y] != 0:
                            self.combat_map.hp_map[self.bomber.x - 1][self.bomber.y] -= 1
                            if self.combat_map.hp_map[self.bomber.x - 1][self.bomber.y] == 0:
                                reward += self.combat_map.value_map[self.bomber.x - 1][self.bomber.y]
                                self.combat_map.value_map[self.bomber.x - 1][self.bomber.y] = 0
                                self.combat_map.is_passable_map[self.bomber.x - 1][self.bomber.y] = 1
                                del self.combat_map.dict_enemy_base[(self.bomber.x - 1, self.bomber.y)]
                            self.bomber.ammo -= 1
                            reward += 1
                        else:
                            reward = -10000
                elif action == 7:
                    if self.bomber.x < self.combat_map.width - 1:
                        if self.combat_map.value_map[self.bomber.x + 1][self.bomber.y] != 0:
                            self.combat_map.hp_map[self.bomber.x + 1][self.bomber.y] -= 1
                            if self.combat_map.dict_enemy_base[(self.bomber.x + 1, self.bomber.y)].hp == 0:
                                reward += self.combat_map.value_map[self.bomber.x + 1][self.bomber.y]
                                self.combat_map.value_map[self.bomber.x + 1][self.bomber.y] = 0
                                self.combat_map.is_passable_map[self.bomber.x + 1][self.bomber.y] = 1
                                del self.combat_map.dict_enemy_base[(self.bomber.x + 1, self.bomber.y)]
                            self.bomber.ammo -= 1
                            reward += 1
                        else:
                            reward = -10000

        else:
            # 如果在己方基地地块上，补给
            if (self.bomber.x, self.bomber.y) in self.combat_map.dict_friendly_base:
                ammo_need = self.bomber.max_ammo - self.bomber.ammo
                fuel_need = self.bomber.max_fuel - self.bomber.fuel
                if ammo_need > 0:
                    ammo_supply = self.combat_map.ammo_map[(self.bomber.x, self.bomber.y)]
                    if ammo_supply >= ammo_need:
                        self.bomber.ammo = self.bomber.max_ammo
                        self.combat_map.ammo_map[(self.bomber.x, self.bomber.y)] -= ammo_need
                        reward += ammo_need * 0.1
                    else:
                        self.bomber.ammo += ammo_supply
                        self.combat_map.ammo_map[(self.bomber.x, self.bomber.y)] = 0
                        reward += ammo_supply * 0.1
                if fuel_need > 0:
                    fuel_supply = self.combat_map.fuel_map[(self.bomber.x, self.bomber.y)]
                    if fuel_supply >= fuel_need:
                        self.bomber.fuel = self.bomber.max_fuel
                        self.combat_map.fuel_map[(self.bomber.x, self.bomber.y)] -= fuel_need
                        reward += fuel_need * 0.1
                    else:
                        self.bomber.fuel += fuel_supply
                        self.combat_map.fuel_map[(self.bomber.x, self.bomber.y)] = 0
                        reward += fuel_supply * 0.1
            else:
                # 如果不在己方基地地块上，reward为-10000
                reward = -10000
        done = self.combat_map.dict_enemy_base == {} or self.bomber.fuel <= 0
        if flag_moved:
            # 如果移动了，结束回合，施加一个指数级的负奖励
            self.turn += 1
            reward -= math.exp(0.1 * self.turn)-1
        return self.get_observation(), reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.combat_map = copy.deepcopy(self.original_combat_map)
        self.bomber = self.combat_map.dict_bomber[0]
        self.turn = 0
        return self.get_observation()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_observation(self):
        # 返回4个通道的图像，分别是ammo_map, fuel_map, hp_map, value_map
        observation = np.zeros((self.combat_map.width, self.combat_map.height, 4))
        observation[:,:,0] = self.combat_map.ammo_map
        observation[:,:,1] = self.combat_map.fuel_map
        for key in self.combat_map.dict_enemy_base:
            observation[key[0]][key[1]][2] = self.combat_map.dict_enemy_base[key].hp
        for key in self.combat_map.dict_enemy_base:
            observation[key[0]][key[1]][3] = self.combat_map.dict_enemy_base[key].value
        return observation

    def get_reward(self):
        pass

    def get_done(self):
        pass

    def get_info(self):
        pass
