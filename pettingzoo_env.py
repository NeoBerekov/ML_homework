import functools

import numpy as np
from pettingzoo.utils import AECEnv,agent_selector,wrappers
from gymnasium import spaces
from customized_test import api_test
from pettingzoo.classic import tictactoe_v3

import copy

from pettingzoo.utils.env import AgentID

JET_POSITION = 0
JET_NOT_MOVED = 1
JET_FUEL = 2
JET_MISSILE = 3
BLUE_BASE_FUEL = 4
BLUE_BASE_MISSILE = 5
RED_BASE_DEFENSE = 6
RED_BASE_VALUE = 7

ROW = 0
COL = 1
FUEL = 2
MISSILE = 3
CAN_MOVE = 4
MAX_FUEL = 5
MAX_MISSILE = 6

class CustomMilitaryEnv(AECEnv):
    def __init__(self, map_size, blue_bases, red_bases, jets):
        super().__init__()

        self.map_size = map_size  # 地图大小 (rows, cols)
        self.blue_bases = blue_bases  # 蓝方基地信息
        self.red_bases = red_bases  # 红方基地信息
        self.jets = copy.deepcopy(jets) # 战斗机信息
        self.original_jets = copy.deepcopy(jets)
        self.agents = [f"jet_{i}" for i in range(len(jets))]
        self.possible_agents = self.agents[:]
        self.rewards = {agent: 0 for agent in self.agents}


        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=-1.0, high=5000.0, shape=(8,map_size[0],map_size[1]), dtype=np.float32),

                "action_mask": spaces.MultiBinary(11),
            }) for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(11) for agent in self.agents
            # 0: 上, 1: 下, 2: 左, 3: 右, 4: 向上攻击, 5: 向下攻击, 6: 向左攻击, 7: 向右攻击, 8: 补给燃油, 9: 补给弹药，10：无动作
        }

        self.state = np.zeros((8,map_size[0],map_size[1]),dtype=np.float32)  # 初始化环境状态
        self.agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_selector.reset()
        self.turn = 0
        self.time_limit = 1000  # 回合数限制

        # 初始化累积奖励、结束标志和信息字典
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.temp_rewards= {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._reset_environment()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        if agent not in self.observation_spaces:
            raise ValueError(f"Unknown agent: {agent}")
        return self.observation_spaces[agent]

    def _reset_environment(self):
        self.state = np.zeros((8,map_size[0],map_size[1]), dtype=np.float32)
        self.jets = copy.deepcopy(self.original_jets)
        # 设置蓝方基地信息
        for base in self.blue_bases:
            assert 0 <= base[0] < self.map_size[0], "蓝方基地行索引超出范围"
            assert 0 <= base[1] < self.map_size[1], "蓝方基地列索引超出范围"
            self.state[BLUE_BASE_FUEL,base[0],base[1]] = base[2]  # 燃油储备量
            self.state[BLUE_BASE_MISSILE,base[0],base[1]] = base[3] # 弹药储备量

        # 设置红方基地信息
        for base in self.red_bases:
            assert 0 <= base[0] < self.map_size[0], "红方基地行索引超出范围"
            assert 0 <= base[1] < self.map_size[1], "红方基地列索引超出范围"
            self.state[RED_BASE_DEFENSE, base[0], base[1]] = base[4]  # 防御值
            self.state[RED_BASE_VALUE, base[0], base[1]] = base[5] # 基地价值

        # 设置战斗机信息
        for i, jet in enumerate(self.jets):
            assert 0 <= jet[0] < self.map_size[0], "战斗机行索引超出范围"
            assert 0 <= jet[1] < self.map_size[1], "战斗机列索引超出范围"
            self.state[JET_POSITION, jet[0], jet[1]] += 1  # 战斗机位置
            self.state[JET_NOT_MOVED, jet[0], jet[1]] += 1  # 战斗机未移动
            self.state[JET_FUEL, jet[0], jet[1]] += jet[FUEL] # 燃油量
            self.state[JET_MISSILE, jet[0], jet[1]] += jet[MISSILE] # 导弹量

    def _convert_to_dict(self, list_of_rew):
        return dict(zip(self.possible_agents, list_of_rew))
    def reset(self,seed=None,options=None):
        self.turn = 0
        self.agents = self.possible_agents[:]
        self.jets = copy.deepcopy(self.original_jets)
        self._reset_environment()
        self.agent_selector.reinit(self.agents)
        self.agent_selector.reset()
        self.agent_selection = self.agent_selector.reset()

        # 重置累积奖励、结束标志和信息字典
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.temp_rewards = {agent: 0 for agent in self.agents}


    def observe(self, agent):
        observation = {"observation": self.state, "action_mask": np.ones(11, dtype=np.int8)}
        return observation

    def step(self, action):

        if np.all(self.state[RED_BASE_DEFENSE] == 0):
            print("All red bases have been destroyed, you are the winner!")
            # print(self.infos)
            self.terminations = {agent: True for agent in self.agents}
        elif len(self.agents) == 0:
            print("All jets have run out of fuel, you are the loser!")
            self.truncations = {agent: True for agent in self.agents}

        if (self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]):
            action = None
            # print(f"{self.agent_selection} has been terminated or truncated")
            self._was_dead_step(action)

            # else:
            #     # print("All jets have been removed from the map")
            #     self.agent_selection = None
            return

        agent = self.agent_selection
        agent_index = self.agents.index(agent)
        jet = self.jets[agent_index]
        self._cumulative_rewards[agent] = 0
        reward = 0.0



        # 处理动作逻辑（移动、攻击、补给等）
        if action == 0:  # 向上移动
            new_position = (jet[0] - 1, jet[1])
            if self._is_valid_move(new_position,jet):
                self._move(agent=agent,jet=jet,new_position=new_position)
                reward += 1
            else:
                reward -= 100
        elif action == 1:  # 向下移动
            new_position = (jet[0] + 1, jet[1])
            if self._is_valid_move(new_position,jet):
                self._move(agent=agent,jet=jet,new_position=new_position)
                reward += 1
            else:
                reward -= 100
        elif action == 2:  # 向左移动
            new_position = (jet[0], jet[1] - 1,jet)
            if self._is_valid_move(new_position,jet):
                self._move(agent=agent,jet=jet,new_position=new_position)
                reward += 1
            else:
                reward -= 100
        elif action == 3:  # 向右移动
            new_position = (jet[0], jet[1] + 1)
            if self._is_valid_move(new_position,jet):
                self._move(agent, jet, new_position)
                reward += 1
            else:
                reward -= 100
        elif action == 4:  # 向上攻击
            if self._is_valid_attack((jet[0] - 1, jet[1])):
                reward += self._attack(agent, jet, (-1, 0))
            else:
                reward -= 100
        elif action == 5:  # 向下攻击
            if self._is_valid_attack((jet[0] + 1, jet[1])):
                reward += self._attack(agent, jet, (1, 0))
            else:
                reward -= 100
        elif action == 6:  # 向左攻击
            if self._is_valid_attack((jet[0], jet[1] - 1)):
                reward += self._attack(agent, jet, (0, -1))
            else:
                reward -= 100
        elif action == 7:  # 向右攻击
            if self._is_valid_attack((jet[0], jet[1] + 1)):
                reward += self._attack(agent, jet, (0, 1))
            else:
                reward -= 100
        elif action == 8:  # 补给燃油
            if self._is_valid_supply_fuel(jet):
               reward += self._supply_fuel(jet,agent)
            else:
                reward -= 100
        elif action == 9:  # 补给弹药
            if self._is_valid_supply_missiles(jet):
                reward += self._supply_missiles(jet,agent)
            else:
                reward -= 100
        elif action == 10:  # 无动作
            reward = 0

        # self.temp_rewards[agent] += reward
        # if self.agent_selector.is_last():
        #     self.rewards = self.temp_rewards
        #     self.temp_rewards = {agent: 0 for agent in self.agents}


        # # 如果一个战斗机已经移动过，燃油为0，且不在有燃油的基地上，则剔除该战斗机
        # if jet[FUEL] <= 0 and jet[CAN_MOVE] == False and self.state[BLUE_BASE_FUEL, jet[ROW], jet[COL]] <= 0:
        #     self.truncations[agent] = True
        #     # print(f"{agent} has run out of fuel and has been removed from the map")

        # 如果超过回合数限制，则所有战斗机都被剔除
        if self.turn >= self.time_limit:
            self.truncations = {agent: True for agent in self.agents}
            # print("Time limit exceeded, all jets have been removed from the map")

        info = {"turn": self.turn, "action": action, "rew": reward,"fuel": jet[FUEL],"missile": jet[MISSILE]}

        if np.all(self.state[RED_BASE_DEFENSE] == 0):
            # print("All red bases have been destroyed, you are the winner!")
            self.terminations = {agent: True for agent in self.agents}

        elif len(self.agents) == 0:
            # print("All jets have run out of fuel, you are the loser!")
            self.truncations = {agent: True for agent in self.agents}

        # 如果所有战斗机都已经移动过，则进入下一个回合，恢复所有战斗机的移动状态
        if all(jet[4] == False for jet in self.jets):
            # print("All jets have moved, next turn")
            self.turn += 1
            # 使得所有战斗机都能移动
            for jet in self.jets:
                jet[4] = True
                # 在地图上标记所有战斗机都能移动
                self.state[JET_NOT_MOVED,jet[0],jet[1]] += 1

        if agent in self.infos.keys():
            self.infos[agent] = info
        # print(f"available agents: {self.agents}")
        # if(len(self.agents)<len(self.possible_agents)):
        #     print("agent removed")
        # print(f"turn: {self.turn}, agent: {agent}, fuel：{jet[FUEL]},action: {action}, reward: {reward}")
        # print(f"Info: {self.infos}")
        # self.render()
        if len(self.agent_selector.agent_order):
            self.agent_selection = self.agent_selector.next()
        self._clear_rewards()
        self.rewards[agent] = reward
        self._accumulate_rewards()
        self._deads_step_first()




    def _is_valid_move(self, position,jet):
        row = position[0]
        col = position[1]
        # 检查是否越界
        if not (0 <= row < self.map_size[0] and 0 <= col < self.map_size[1]):
            return False
        # 检查是否移动到敌方基地
        elif self.state[RED_BASE_DEFENSE,row,col] > 0:
            return False
        # 检查还有没有燃油
        elif jet[2] <= 0:
            return False
        # 检查该回合内是否已经移动过
        elif not jet[4]:
            return False
        return True

    def _move(self, agent, jet, new_position):
        row = new_position[0]
        col = new_position[1]
        # 为了防止战斗机重叠的时候出现问题，用减法逻辑更新战斗机位置
        self.state[JET_POSITION, jet[0], jet[1]] -= 1
        self.state[JET_POSITION, row, col] += 1
        self.state[JET_NOT_MOVED, jet[0], jet[1]] -= 1
        self.state[JET_FUEL, jet[0], jet[1]] -= jet[FUEL]
        self.state[JET_FUEL, row, col] += jet[FUEL]-1
        self.state[JET_MISSILE, jet[0], jet[1]] -= jet[MISSILE]
        self.state[JET_MISSILE, row, col] += jet[MISSILE]
        self.jets[self.agents.index(agent)] = [row, col, jet[FUEL]-1, jet[MISSILE], False, jet[MAX_FUEL], jet[MAX_MISSILE]]


    def _is_valid_attack(self, target_position):
        row, col = target_position
        # 检查目标位置是否在地图边界内
        if not (0 <= row < self.map_size[0] and 0 <= col < self.map_size[1]):
            return False
        # 检查目标位置是否为敌方基地
        if self.state[RED_BASE_DEFENSE,row,col] > 0:
            return True
        return False

    def _attack(self, agent, jet, direction):
        target_position = (jet[0] + direction[0], jet[1] + direction[1])
        row, col = target_position
        base_value = 0
        missiles_fired = min(jet[3], self.state[RED_BASE_DEFENSE,row,col])
        self.state[RED_BASE_DEFENSE,row,col] -= missiles_fired
        self.state[JET_MISSILE,jet[0],jet[1]] -= missiles_fired
        self.jets[self.agents.index(agent)][3] -= missiles_fired
        if self.state[RED_BASE_DEFENSE,row,col] == 0:
            base_value = self.state[RED_BASE_VALUE,row,col]
            self.state[RED_BASE_VALUE,row,col] = 0
        return base_value + missiles_fired/10



    def _is_valid_supply_fuel(self, jet):
        # 检查战斗机是否在己方基地
        if (any((jet[0], jet[1]) == (base[0], base[1]) for base in self.blue_bases)
                and self.state[BLUE_BASE_MISSILE,jet[0],jet[1]] > 0):
            return True
        return False

    def _supply_fuel(self, jet,agent):
        # 假设补给指令已合法性检查
        base = jet[0], jet[1]
        fuel_needed = jet[5]-jet[2]  # 需要的燃油量
        fuel_amount = min(self.state[BLUE_BASE_FUEL,base[0],base[1]]
                          ,fuel_needed)  # 实际加油量，不能超过基地储备
        self.state[JET_FUEL,base[0],base[1]] += fuel_amount
        self.jets[self.agents.index(agent)][2] += fuel_amount
        self.state[BLUE_BASE_FUEL,base[0],base[1]] -= fuel_amount
        return fuel_amount/10 # as reward

    def _is_valid_supply_missiles(self, jet):
        # 检查战斗机是否在己方基地
        if (any((jet[0], jet[1]) == (base[0], base[1]) for base in self.blue_bases)
                and self.state[BLUE_BASE_MISSILE,jet[0],jet[1]] > 0):
            return True
        return False

    def _supply_missiles(self, jet,agent):
        # 假设补给指令已合法性检查
        base_location = jet[0], jet[1]
        missiles_needed = jet[6] - jet[3]  # 需要的弹药量
        missile_amount = min(self.state[BLUE_BASE_MISSILE,base_location[0],base_location[1]]
                             , missiles_needed)  # 实际装弹量，不能超过基地储备
        self.state[JET_MISSILE, jet[0], jet[1],] += missile_amount
        self.jets[self.agents.index(agent)][3] += missile_amount
        self.state[BLUE_BASE_MISSILE, base_location[0],base_location[1]] -= missile_amount
        return missile_amount/10 # as reward

    def render(self):
        # 渲染逻辑实现，在控制台打印一个简单的地图
        # 蓝方基地用字母B表示，红方基地用字母R表示，战斗机用字母J表示
        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                if self.state[JET_POSITION,row,col] > 0:
                    print("J", end=" ")
                elif self.state[RED_BASE_DEFENSE,row,col] > 0:
                    print("R", end=" ")
                elif self.state[BLUE_BASE_FUEL,row,col] > 0:
                    print("B", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()


    def close(self):
        pass


# 示例环境配置
map_size = (10, 10)  # 地图大小
blue_bases = [
    (0, 6, 1000, 200, 5, 50),
    (3, 7, 800, 150, 4, 40),
    (4, 6, 1500, 250, 6, 60),
    (4, 7, 1200, 180, 5, 45),
    (4, 8, 900, 100, 3, 30)
]
red_bases = [
    (1, 2, 1000, 200, 5, 50),
    (2, 2, 800, 150, 4, 40),
    (2, 3, 1500, 250, 6, 60),
    (3, 1, 1200, 180, 5, 45),
    (5, 3, 900, 100, 3, 30)
]
jets = [
    # (row, col, fuel, missile,can_move, max_fuel, max_missile)
    [1, 1, 100, 50,True,1000,500],
    [4, 7, 80, 30,True,800,300],
    [6, 6, 90, 40,True,900,400]
]

for i in range(100):
    print("Episode: ", i)
    # 创建环境
    env = CustomMilitaryEnv(map_size, blue_bases, red_bases, jets)
    # 测试环境API
    api_test(env)


