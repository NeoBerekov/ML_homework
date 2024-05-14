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
from environment import Base, Bomber, CombatMap

class BomberBaseEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,combat_map):
        super(BomberBaseEnv, self).__init__()
        self.action_space = spaces.Discrete(9)
        self.combat_map = combat_map
        self.bomber = combat_map.dict_bomber[0]
        self.original_combat_map = combat_map
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
                if self.bomber.y > 0 and self.combat_map.is_passable_map[self.bomber.x][self.bomber.y-1] == 1:
                    self.bomber.y -= 1
                    self.bomber.fuel -= 1
                    self.turn += 1
                    flag_moved = True
                else:
                    reward = -10000
            elif action == 1:
                if self.bomber.y < self.combat_map.height - 1 and self.combat_map.is_passable_map[self.bomber.x][self.bomber.y+1] == 1:
                    self.bomber.y += 1
                    self.bomber.fuel -= 1
                    self.turn += 1
                    flag_moved = True
                else:
                    reward = -10000
            elif action == 2:
                if self.bomber.x > 0 and self.combat_map.is_passable_map[self.bomber.x-1][self.bomber.y] == 1:
                    self.bomber.x -= 1
                    self.bomber.fuel -= 1
                    self.turn += 1
                    flag_moved = True
                else:
                    reward = -10000
            elif action == 3:
                if self.bomber.x < self.combat_map.width - 1 and self.combat_map.is_passable_map[self.bomber.x+1][self.bomber.y] == 1:
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
                        if (self.bomber.x, self.bomber.y - 1) in self.combat_map.dict_enemy_base:
                            self.combat_map.dict_enemy_base[(self.bomber.x, self.bomber.y - 1)].hp -= 1
                            if self.combat_map.dict_enemy_base[(self.bomber.x, self.bomber.y - 1)].hp == 0:
                                reward += self.combat_map.dict_enemy_base[(self.bomber.x, self.bomber.y - 1)].value
                                del self.combat_map.dict_enemy_base[(self.bomber.x, self.bomber.y - 1)]
                            self.bomber.ammo -= 1
                            reward += 1
                        else:
                            reward = -10000
                    else:
                        reward = -10000
                elif action == 5:
                    if self.bomber.y < self.combat_map.height - 1:
                        if (self.bomber.x, self.bomber.y + 1) in self.combat_map.dict_enemy_base:
                            self.combat_map.dict_enemy_base[(self.bomber.x, self.bomber.y + 1)].hp -= 1
                            if self.combat_map.dict_enemy_base[(self.bomber.x, self.bomber.y + 1)].hp == 0:
                                reward += self.combat_map.dict_enemy_base[(self.bomber.x, self.bomber.y + 1)].value
                                del self.combat_map.dict_enemy_base[(self.bomber.x, self.bomber.y + 1)]
                            self.bomber.ammo -= 1
                            reward += 1
                        else:
                            reward = -10000
                    else:
                        reward = -10000
                elif action == 6:
                    if self.bomber.x > 0:
                        if (self.bomber.x - 1, self.bomber.y) in self.combat_map.dict_enemy_base:
                            self.combat_map.dict_enemy_base[(self.bomber.x - 1, self.bomber.y)].hp -= 1
                            if self.combat_map.dict_enemy_base[(self.bomber.x - 1, self.bomber.y)].hp == 0:
                                reward += self.combat_map.dict_enemy_base[(self.bomber.x - 1, self.bomber.y)].value
                                del self.combat_map.dict_enemy_base[(self.bomber.x - 1, self.bomber.y)]
                            self.bomber.ammo -= 1
                            reward += 1
                        else:
                            reward = -10000
                elif action == 7:
                    if self.bomber.x < self.combat_map.width - 1:
                        if (self.bomber.x + 1, self.bomber.y) in self.combat_map.dict_enemy_base:
                            self.combat_map.dict_enemy_base[(self.bomber.x + 1, self.bomber.y)].hp -= 1
                            if self.combat_map.dict_enemy_base[(self.bomber.x + 1, self.bomber.y)].hp == 0:
                                reward += self.combat_map.dict_enemy_base[(self.bomber.x + 1, self.bomber.y)].value
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
                    ammo_supply = self.combat_map.dict_friendly_base[(self.bomber.x, self.bomber.y)].ammo
                    if ammo_supply >= ammo_need:
                        self.bomber.ammo = self.bomber.max_ammo
                        self.combat_map.dict_friendly_base[(self.bomber.x, self.bomber.y)].ammo -= ammo_need
                        reward += ammo_need * 0.1
                    else:
                        self.bomber.ammo += ammo_supply
                        self.combat_map.dict_friendly_base[(self.bomber.x, self.bomber.y)].ammo = 0
                        reward += ammo_supply * 0.1
                if fuel_need > 0:
                    fuel_supply = self.combat_map.dict_friendly_base[(self.bomber.x, self.bomber.y)].fuel
                    if fuel_supply >= fuel_need:
                        self.bomber.fuel = self.bomber.max_fuel
                        self.combat_map.dict_friendly_base[(self.bomber.x, self.bomber.y)].fuel -= fuel_need
                        reward += fuel_need * 0.1
                    else:
                        self.bomber.fuel += fuel_supply
                        self.combat_map.dict_friendly_base[(self.bomber.x, self.bomber.y)].fuel = 0
                        reward += fuel_supply * 0.1
            else:
                # 如果不在己方基地地块上，reward为-10000
                reward = -10000
        done = self.combat_map.dict_enemy_base == {} or self.bomber.fuel <= 0
        if flag_moved:
            # 如果移动了，结束回合，施加一个指数级的负奖励
            self.turn += 1
            reward -= math.exp(0.1 * self.turn)
        return self.get_observation(), reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.__init__(self, self.original_combat_map)
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def get_observation(self):
        pass

    def get_reward(self):
        pass

    def get_done(self):
        pass

    def get_info(self):
        pass