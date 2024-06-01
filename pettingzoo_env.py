import functools

import numpy as np
import random
from pettingzoo.utils import AECEnv,agent_selector,wrappers
from gymnasium import spaces
from customized_test import api_test

import copy

# 直接点运行能跑100轮api测试

# 定义地图深度对应的信息
JET_POSITION = 0
JET_NOT_MOVED = 1
JET_FUEL = 2
JET_MISSILE = 3
BLUE_BASE_FUEL = 4
BLUE_BASE_MISSILE = 5
RED_BASE_DEFENSE = 6
RED_BASE_VALUE = 7

# 定义战斗机信息对应的索引
ROW = 0
COL = 1
FUEL = 2
MISSILE = 3
CAN_MOVE = 4
MAX_FUEL = 5
MAX_MISSILE = 6

# 0: 上, 1: 下, 2: 左, 3: 右, 4: 向上攻击, 5: 向下攻击, 6: 向左攻击, 7: 向右攻击, 8: 补给燃油, 9: 补给弹药，10：无动作
ACT_MOVE_UP = 0
ACT_MOVE_DOWN = 1
ACT_MOVE_LEFT = 2
ACT_MOVE_RIGHT = 3
ACT_ATTACK_UP = 4
ACT_ATTACK_DOWN = 5
ACT_ATTACK_LEFT = 6
ACT_ATTACK_RIGHT = 7
ACT_SUPPLY_FUEL = 8
ACT_SUPPLY_MISSILES = 9
ACT_NO_OP = 10

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


        # self.observation_spaces = {
        #     agent: spaces.Dict({
        #         "observation": spaces.Box(low=-0.001, high=5000.0, shape=(8,map_size[0],map_size[1]), dtype=np.float32),
        #
        #         "action_mask": spaces.MultiBinary(11),
        #
        #         "position": spaces.Tuple((spaces.Discrete(map_size[0]), spaces.Discrete(map_size[1]))),
        #
        #         "fuel": spaces.Box(low=-0.001, high=5000, shape=(1,), dtype=np.float32),
        #
        #         "missile": spaces.Box(low=-0.001, high=5000, shape=(1,), dtype=np.float32)
        #     }) for agent in self.agents
        # }
        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Dict({
                    "position": spaces.Tuple((
                        spaces.Discrete(map_size[0]),
                        spaces.Discrete(map_size[1])
                    )),
                    "fuel": spaces.Box(low=-0.001, high=5000, shape=(1,), dtype=np.float32),
                    "missile": spaces.Box(low=-0.001, high=5000, shape=(1,), dtype=np.float32),
                    "map_obs": spaces.Box(low=-0.001, high=5000.0, shape=(8, map_size[0], map_size[1]),
                                                     dtype=np.float32)
                }),
                "action_mask": spaces.MultiBinary(11)
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
        self.state = np.zeros((8,self.map_size[0],self.map_size[1]), dtype=np.float32)
        self.jets = copy.deepcopy(self.original_jets)
        # 设置蓝方基地信息
        for base in self.blue_bases:
            assert 0 <= base[0] < self.map_size[0], "蓝方基地行索引超出范围"
            assert 0 <= base[1] < self.map_size[1], "蓝方基地列索引超出范围"
            self.state[BLUE_BASE_FUEL,base[0],base[1]] += base[2]  # 燃油储备量
            self.state[BLUE_BASE_MISSILE,base[0],base[1]] += base[3] # 弹药储备量

        # 设置红方基地信息
        for base in self.red_bases:
            assert 0 <= base[0] < self.map_size[0], "红方基地行索引超出范围"
            assert 0 <= base[1] < self.map_size[1], "红方基地列索引超出范围"
            self.state[RED_BASE_DEFENSE, base[0], base[1]] = +base[4]  # 防御值
            self.state[RED_BASE_VALUE, base[0], base[1]] = +base[5] # 基地价值

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
        # print()
        # print("reset")
        # print()
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
        # 目前是软控制飞机的合法动作，要是想硬控制可以调action_mask，在main里的TerminateIllegalWrapper会根据这个mask来判断是否合法
        observation = {"observation": {
            "position": tuple([self.jets[self.agents.index(agent)][ROW], self.jets[self.agents.index(agent)][COL]]),
            "fuel": np.array([self.jets[self.agents.index(agent)][FUEL]]).astype(np.float32),
            "missile": np.array([self.jets[self.agents.index(agent)][MISSILE]]).astype(np.float32),
            # "fuel": np.float32(self.jets[self.agents.index(agent)][FUEL]),
            # "missile": np.float32(self.jets[self.agents.index(agent)][MISSILE]),
            "map_obs": self.state,
        },
                       "action_mask": np.ones(11, dtype=np.int8),
                       }
        return observation

    def step(self, action):
        # print(f"turn: {self.turn}, agent: {self.agent_selection}, action: {action}")
        if (self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]):
            # 如果当前agent已经终止或被截断，则直接跳过
            action = None
            # print(f"{self.agent_selection} has been terminated or truncated")
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        agent_index = self.agents.index(agent)
        jet = self.jets[agent_index]
        self._cumulative_rewards[agent] = 0
        reward = 0.0



        # 处理动作逻辑（移动、攻击、补给等）
        if action == 0:  # 向上移动
            new_position = (jet[0] - 1, jet[1])
            if self._is_valid_move(new_position,jet): # 软禁止飞机移动到敌方基地
                reward += self._move(agent=agent,jet=jet,new_position=new_position)
            else:
                reward -= 100
        elif action == 1:  # 向下移动
            new_position = (jet[0] + 1, jet[1])
            if self._is_valid_move(new_position,jet):
                reward += self._move(agent=agent,jet=jet,new_position=new_position)
            else:
                reward -= 100
        elif action == 2:  # 向左移动
            new_position = (jet[0], jet[1] - 1,jet)
            if self._is_valid_move(new_position,jet):
                reward += self._move(agent=agent,jet=jet,new_position=new_position)
            else:
                reward -= 100
        elif action == 3:  # 向右移动
            new_position = (jet[0], jet[1] + 1)
            if self._is_valid_move(new_position,jet):
                reward += self._move(agent=agent,jet=jet,new_position=new_position)
            else:
                reward -= 100
        elif action == 4:  # 向上攻击
            if self._is_valid_attack((jet[0] - 1, jet[1])): # 软禁止飞机攻击非敌方地块
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
            if self._is_valid_supply_fuel(jet): # 软禁止飞机在无燃油地块加油
               reward += self._supply_fuel(jet,agent)
            else:
                reward -= 100
        elif action == 9:  # 补给弹药
            if self._is_valid_supply_missiles(jet): # 软禁止飞机在无弹药地块补给弹药
                reward += self._supply_missiles(jet,agent)
            else:
                reward -= 100
        elif action == 10:  # 无动作
            if jet[FUEL] == 0: # 没油导致的不动，加负奖励
                reward -= 10

        # 如果超过回合数限制，则所有战斗机都被剔除
        if self.turn >= self.time_limit:
            self.truncations = {agent: True for agent in self.agents}
            # print("Time limit exceeded, all jets have been removed from the map")

        info = {"turn": self.turn, "action": action, "rew": reward,"fuel": jet[FUEL],"missile": jet[MISSILE]}

        if np.all(self.state[RED_BASE_DEFENSE] == 0):
            print(f"Turn:{self.turn},All red bases have been destroyed, you are the winner!")
            print("last hit by: ", agent)
            # self.render()
            self.terminations = {agent: True for agent in self.agents}

        elif all(jet[FUEL] == 0
                 and jet[CAN_MOVE] == False
                 and self.state[BLUE_BASE_FUEL,jet[ROW],jet[COL]] <= 0
                 for jet in self.jets): # 所有战斗机都已经移动过且燃油耗尽，而且无法补给燃油，游戏结束
            # print(f"Turn:{self.turn},All jets have run out of fuel, you are the loser!")
            # self.render()
            self.truncations = {agent: True for agent in self.agents}

        # 如果所有战斗机都已经移动过，则进入下一个回合，恢复所有战斗机的移动状态
        if all(jet[CAN_MOVE] == False for jet in self.jets):
            # print("All jets have moved, next turn")
            self.turn += 1
            # 使得所有战斗机都能移动
            for jet in self.jets:
                if jet[FUEL] > 0:
                    jet[CAN_MOVE] = True
                    # 在地图上标记所有战斗机都能移动
                    self.state[JET_NOT_MOVED, jet[0], jet[1]] += 1

        if agent in self.infos.keys():
            self.infos[agent] = info
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
        if jet[FUEL] <= 0:
            # 如果燃油为0，不执行移动操作，并将战斗机的移动状态设置为False
            self.jets[self.agents.index(agent)][CAN_MOVE] = False
            self.state[JET_NOT_MOVED, jet[0], jet[1]] -= 1
            return -100
        else:
            row = new_position[0]
            col = new_position[1]
            # 为了防止战斗机重叠的时候出现问题，用减法逻辑更新战斗机位置
            self.state[JET_POSITION, jet[0], jet[1]] -= 1
            self.state[JET_POSITION, row, col] += 1
            self.state[JET_NOT_MOVED, jet[0], jet[1]] -= 1
            self.state[JET_FUEL, jet[0], jet[1]] -= jet[FUEL]
            self.state[JET_FUEL, row, col] += jet[FUEL] - 1
            self.state[JET_MISSILE, jet[0], jet[1]] -= jet[MISSILE]
            self.state[JET_MISSILE, row, col] += jet[MISSILE]
            self.jets[self.agents.index(agent)] = [row, col, jet[FUEL] - 1, jet[MISSILE], False, jet[MAX_FUEL],
                                                   jet[MAX_MISSILE]]
            return 1


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
        if jet[MISSILE]<=0:
            return -100
        else:
            target_position = (jet[0] + direction[0], jet[1] + direction[1])
            row, col = target_position
            base_value = 0
            missiles_fired = min(jet[3], self.state[RED_BASE_DEFENSE, row, col])
            self.state[RED_BASE_DEFENSE, row, col] -= missiles_fired
            self.state[JET_MISSILE, jet[0], jet[1]] -= missiles_fired
            self.jets[self.agents.index(agent)][3] -= missiles_fired
            if self.state[RED_BASE_DEFENSE, row, col] == 0:
                base_value = self.state[RED_BASE_VALUE, row, col]
                self.state[RED_BASE_VALUE, row, col] = 0
            return base_value + missiles_fired / 10



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


def read_data_file(file_path):
    # 记得把testcase里的那个地图删掉，只留数字信息
    # 我把处理后的testcase扔到了testcase文件夹里
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 读取地图大小
    map_size = tuple(map(int, lines[0].strip().split()))

    # 读取蓝方基地信息
    blue_bases = []
    blue_bases_count = int(lines[1].strip())
    index = 2
    for _ in range(blue_bases_count):
        row, col, fuel, missile, defense, value = map(int, lines[index].strip().split())
        blue_bases.append((row, col, fuel, missile, defense, value))
        index += 1

    # 读取红方基地信息
    red_bases = []
    red_bases_count = int(lines[index].strip())
    index += 1
    for _ in range(red_bases_count):
        row, col, fuel, missile, defense, value = map(int, lines[index].strip().split())
        red_bases.append((row, col, fuel, missile, defense, value))
        index += 1

    # 读取战斗机信息
    jets = []
    jets_count = int(lines[index].strip())
    index += 1
    for _ in range(jets_count):
        row, col, max_fuel, max_missile = map(int, lines[index].strip().split())
        fuel, missile = 0, 0
        can_move = fuel!=0
        jets.append([row, col, fuel, missile, can_move, max_fuel, max_missile])
        index += 1

    return map_size, blue_bases, red_bases, jets


if __name__ == '__main__':
    map_size, blue_bases, red_bases, jets = read_data_file("testcase/test1.txt")
    # print(map_size)
    # print(blue_bases)
    # print(red_bases)
    # print(jets)

    for i in range(100):
        random.seed(i)
        np.random.seed(i)
        print("Episode: ", i)
        # 创建环境
        env = CustomMilitaryEnv(map_size, blue_bases, red_bases, jets)
        # 测试环境API
        if not api_test(env):
            print("Failed")
            break


