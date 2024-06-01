import gymnasium as gym
import numpy as np
import copy

# 定义地图深度对应的信息
JET_POSITION = 0
JET_NOT_MOVED = 1
JET_FUEL = 2
JET_MISSILE = 3
BLUE_BASE_FUEL = 4
BLUE_BASE_MISSILE = 5
RED_BASE_DEFENSE = 6
RED_BASE_VALUE = 7
BOUNDARY_GRADIENT = 8

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


class CombatEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_size, blue_bases, red_bases, jets,local_obs_window=5,max_turns=1000):
        super().__init__()

        self.map_size = map_size  # 地图大小 (rows, cols)
        self.blue_bases = blue_bases  # 蓝方基地信息
        self.red_bases = red_bases  # 红方基地信息
        self.local_obs_window = local_obs_window  # 局部观察窗口大小,n*n
        self.num_red_bases = len(red_bases)  # 红方基地数量
        self.destroyed_red_bases = 0  # 被摧毁的红方基地数量
        self.no_op_steps = 0  # 无动作步数
        self.jets = copy.deepcopy(jets) # 战斗机信息
        self.original_jets = copy.deepcopy(jets)
        self.current_jet = 0  # 当前战斗机索引
        self.turn = 0  # 当前回合数
        self.turn_step = 0  # 当前回合步数
        self.total_step = 0  # 总步数
        self.terminated = False  # 游戏是否结束
        self.truncated = False  # 是否截断游戏
        # self.reward = 0
        self.global_state = None
        self.max_turns = max_turns  # 最大回合数
        self.illegal_moves = 0  # 非法移动次数

        self.action_space = gym.spaces.Discrete(11)  # 动作空间

        self.observation_space = gym.spaces.Dict({
            'local_obs': gym.spaces.Box(low=-1, high=5000, shape=(9, local_obs_window, local_obs_window), dtype=np.int16),
            'global_obs': gym.spaces.Box(low=0, high=5000, shape=(9, map_size[0],map_size[1]), dtype=np.int16),
            'action_mask': gym.spaces.MultiBinary(11)
        }) # 观察空间

        self.reset()
    def create_action_mask(self):
        action_mask = np.zeros(11, dtype=np.int16)
        jet = self.jets[self.current_jet]
        if jet[FUEL] > 0 and jet[CAN_MOVE]:
            # 如果战斗机燃油量大于0,本回合未移动且对应的方向上没有红方基地，则可以移动
            if jet[ROW] > 0 and self.global_state[RED_BASE_DEFENSE, jet[ROW] - 1, jet[COL]] == 0:
                action_mask[ACT_MOVE_UP] = 1
            if jet[ROW] < self.map_size[0] - 1 and self.global_state[RED_BASE_DEFENSE, jet[ROW] + 1, jet[COL]] == 0:
                action_mask[ACT_MOVE_DOWN] = 1
            if jet[COL] > 0 and self.global_state[RED_BASE_DEFENSE, jet[ROW], jet[COL] - 1] == 0:
                action_mask[ACT_MOVE_LEFT] = 1
            if jet[COL] < self.map_size[1] - 1 and self.global_state[RED_BASE_DEFENSE, jet[ROW], jet[COL] + 1] == 0:
                action_mask[ACT_MOVE_RIGHT] = 1
        if jet[MISSILE] > 0:
            # 如果战斗机导弹量大于0且对应的方向上有红方基地，则可以攻击
            if jet[ROW] > 0 and self.global_state[RED_BASE_DEFENSE, jet[ROW] - 1, jet[COL]] > 0:
                action_mask[ACT_ATTACK_UP] = 1
            if jet[ROW] < self.map_size[0] - 1 and self.global_state[RED_BASE_DEFENSE, jet[ROW] + 1, jet[COL]] > 0:
                action_mask[ACT_ATTACK_DOWN] = 1
            if jet[COL] > 0 and self.global_state[RED_BASE_DEFENSE, jet[ROW], jet[COL] - 1] > 0:
                action_mask[ACT_ATTACK_LEFT] = 1
            if jet[COL] < self.map_size[1] - 1 and self.global_state[RED_BASE_DEFENSE, jet[ROW], jet[COL] + 1] > 0:
                action_mask[ACT_ATTACK_RIGHT] = 1
        if self.global_state[BLUE_BASE_FUEL, jet[ROW], jet[COL]] > 0:
            action_mask[ACT_SUPPLY_FUEL] = 1
        if self.global_state[BLUE_BASE_MISSILE, jet[ROW], jet[COL]] > 0:
            action_mask[ACT_SUPPLY_MISSILES] = 1
        action_mask[ACT_NO_OP] = 0
        return action_mask
    def create_gradient_map(self,max_distance):
        """
        创建一个梯度地图，其中边界的距离max_distance内具有梯度值，边界处为1，向内逐渐减小至0。
        map_size: 地图的尺寸，形式为(rows, cols)
        max_distance: 边界梯度的最大影响距离
        """
        gradient_map = np.zeros(self.map_size)

        rows, cols = self.map_size
        for x in range(rows):
            for y in range(cols):
                min_dist = min(x, rows - 1 - x, y, cols - 1 - y)
                if min_dist < max_distance:
                    gradient_map[x, y] = (max_distance - min_dist) / max_distance

        return gradient_map

    def reset(self,seed=114514,options=None):
        super().reset(seed=seed,options=options)

        self.global_state = np.zeros((9, self.map_size[0], self.map_size[1]), dtype=np.int16)
        self.jets = copy.deepcopy(self.original_jets)

        for base in self.blue_bases:
            assert 0 <= base[0] < self.map_size[0], "蓝方基地行索引超出范围"
            assert 0 <= base[1] < self.map_size[1], "蓝方基地列索引超出范围"
            self.global_state[BLUE_BASE_FUEL,base[0],base[1]] += base[2]  # 燃油储备量
            self.global_state[BLUE_BASE_MISSILE,base[0],base[1]] += base[3] # 弹药储备量

        # 设置红方基地信息
        for base in self.red_bases:
            assert 0 <= base[0] < self.map_size[0], "红方基地行索引超出范围"
            assert 0 <= base[1] < self.map_size[1], "红方基地列索引超出范围"
            self.global_state[RED_BASE_DEFENSE, base[0], base[1]] = +base[4]  # 防御值
            self.global_state[RED_BASE_VALUE, base[0], base[1]] = +base[5] # 基地价值

        # 设置战斗机信息
        for i, jet in enumerate(self.jets):
            assert 0 <= jet[0] < self.map_size[0], "战斗机行索引超出范围"
            assert 0 <= jet[1] < self.map_size[1], "战斗机列索引超出范围"
            self.global_state[JET_POSITION, jet[0], jet[1]] += 1  # 战斗机位置
            if jet[FUEL] > 0:
                self.global_state[JET_NOT_MOVED, jet[0], jet[1]] += 1
            self.global_state[JET_FUEL, jet[0], jet[1]] += jet[FUEL] # 燃油量
            self.global_state[JET_MISSILE, jet[0], jet[1]] += jet[MISSILE] # 导弹量

        # 创建梯度地图
        self.global_state[BOUNDARY_GRADIENT] = self.create_gradient_map(3)


        self.terminated = False  # 游戏是否结束
        self.truncated = False  # 是否截断游戏
        # self.reward = 0
        self.turn = 0
        self.turn_step = 0
        self.total_step = 0
        self.destroyed_red_bases = 0
        self.num_red_bases = len(self.red_bases)
        self.current_jet = 0
        self.illegal_moves = 0
        self.no_op_steps = 0


        return {
            "local_obs":
                self.extract_local_observation(
                    self.global_state,
                    self.jets[self.current_jet][0:2],
                    self.local_obs_window,
                    8),
            "global_obs":
                self.global_state,
            'action_mask':
                self.create_action_mask()
                }

    def extract_local_observation(self,full_map, position, window_size, gradient_depth):
        """
        从完整的地图中提取局部观察区域。
        full_map: 完整的地图，假设其形状为(num_depths, height, width)
        position: 智能体的位置，形式为(x, y)
        window_size: 观察窗口的大小，n*n
        gradient_depth: 指定哪一层是梯度层
        """
        num_depths, height, width = full_map.shape
        half_window = window_size // 2

        # 计算局部观察区域的起始和结束索引
        start_x = position[0] - half_window
        end_x = position[0] + half_window + 1
        start_y = position[1] - half_window
        end_y = position[1] + half_window + 1

        # 创建一个填充值为-1的数组，大小与窗口相同
        local_obs = np.full((num_depths, window_size, window_size), -1, dtype=np.float32)

        # 计算裁剪后的实际起始和结束索引
        clip_start_x = max(start_x, 0)
        clip_end_x = min(end_x, height)
        clip_start_y = max(start_y, 0)
        clip_end_y = min(end_y, width)

        # 计算填充到local_obs中的索引位置
        pad_start_x = clip_start_x - start_x
        pad_end_x = pad_start_x + (clip_end_x - clip_start_x)
        pad_start_y = clip_start_y - start_y
        pad_end_y = pad_start_y + (clip_end_y - clip_start_y)

        # 填充局部观察数组
        local_obs[:, pad_start_x:pad_end_x, pad_start_y:pad_end_y] = full_map[:, clip_start_x:clip_end_x,
                                                                     clip_start_y:clip_end_y]

        # 如果有超出边界的部分，特殊处理梯度层
        if gradient_depth is not None:
            local_obs[gradient_depth][local_obs[gradient_depth] == -1] = 1

        return np.array(local_obs)

    def check_termination(self):
        # 检查是否所有红方基地血量为0
        return self.destroyed_red_bases >= self.num_red_bases

    def check_truncation(self):
        # 检查是否超出回合数或者所有战斗机都燃油为0且不在有燃油补给的基地上
        return self.turn >= self.max_turns or \
               (all(jet[FUEL] == 0 for jet in self.jets) and
                all(self.global_state[BLUE_BASE_FUEL, jet[ROW], jet[COL]] <= 0 for jet in self.jets))


    def move(self,jet_index, new_pos):
        jet = self.jets[jet_index]
        # 检查新位置是否合法
        if new_pos[0] < 0 or new_pos[0] >= self.map_size[0] or new_pos[1] < 0 or new_pos[1] >= self.map_size[1]:
            self.illegal_moves += 1
            return -100
        # 检查新位置是否有红方基地
        if self.global_state[RED_BASE_DEFENSE, new_pos[0], new_pos[1]] > 0:
            self.illegal_moves += 1
            return -100
        # 检查燃油是否足够
        if jet[CAN_MOVE] == False:
            self.illegal_moves += 1
            return -100
        if jet[FUEL] <= 0 and jet[CAN_MOVE] == True:
            self.illegal_moves += 1
            jet[CAN_MOVE] = False
            self.global_state[JET_NOT_MOVED, jet[ROW], jet[COL]] -= 1
            return -100
        else:
            self.global_state[JET_NOT_MOVED, jet[ROW], jet[COL]] -= 1
            # 更新战斗机位置
            self.global_state[JET_POSITION, new_pos[0], new_pos[1]] += 1
            self.global_state[JET_FUEL, new_pos[0], new_pos[1]] += jet[FUEL] - 1
            self.global_state[JET_MISSILE, new_pos[0], new_pos[1]] += jet[MISSILE]
            self.global_state[JET_POSITION, jet[ROW], jet[COL]] -= 1
            self.global_state[JET_NOT_MOVED, jet[ROW], jet[COL]] -= 1
            self.global_state[JET_FUEL, jet[ROW], jet[COL]] -= jet[FUEL]
            self.global_state[JET_MISSILE, jet[ROW], jet[COL]] -= jet[MISSILE]
            jet[ROW] = new_pos[0]
            jet[COL] = new_pos[1]
            jet[FUEL] -= 1
            jet[CAN_MOVE] = False
            return -1

    def attack(self,jet_index, position):
        jet = self.jets[jet_index]
        reward = 0
        # 检查导弹数量是否足够
        if jet[MISSILE] <= 0:
            self.illegal_moves += 1
            return -100
        # 检查攻击位置是否合法
        if position[0] < 0 or position[0] >= self.map_size[0] or position[1] < 0 or position[1] >= self.map_size[1]:
            self.illegal_moves += 1
            return -100
        # 检查攻击位置是否有红方基地
        if self.global_state[RED_BASE_DEFENSE, position[0], position[1]] <= 0:
            self.illegal_moves += 1
            return -100
        # 计算使用导弹数量
        missile_used = min(jet[MISSILE], self.global_state[RED_BASE_DEFENSE, position[0], position[1]])
        # 更新导弹数量
        jet[MISSILE] -= missile_used
        self.global_state[JET_MISSILE, jet[ROW], jet[COL]] -= missile_used
        self.global_state[RED_BASE_DEFENSE, position[0], position[1]] -= missile_used
        reward += missile_used
        # 如果摧毁了基地，更新基地价值，附加对应奖励
        if self.global_state[RED_BASE_DEFENSE, position[0], position[1]] <= 0:
            self.destroyed_red_bases += 1
            reward += 10 * self.global_state[RED_BASE_VALUE, position[0], position[1]]
            self.global_state[RED_BASE_VALUE, position[0], position[1]] = 0
            print(f"Turn {self.turn}")
            print("Red base at ({}, {}) has been destroyed".format(position[0], position[1]))
        return reward

    def supply(self,jet_index, position, supply_type):
        # ACT_SUPPLY_FUEL: 补给燃油, ACT_SUPPLY_MISSILE: 补给导弹
        jet = self.jets[jet_index]
        reward = 0
        # 检查位置是否合法
        if position[0] < 0 or position[0] >= self.map_size[0] or position[1] < 0 or position[1] >= self.map_size[1]:
            self.illegal_moves += 1
            return -100
        # 检查位置是否有蓝方基地
        if self.global_state[BLUE_BASE_FUEL, position[0], position[1]] <= 0 and supply_type == ACT_SUPPLY_FUEL:
            self.illegal_moves += 1
            return -100
        if self.global_state[BLUE_BASE_MISSILE, position[0], position[1]] <= 0 and supply_type == ACT_SUPPLY_MISSILES:
            self.illegal_moves += 1
            return -100
        # 补给燃油
        if supply_type == ACT_SUPPLY_FUEL:
            # 计算补给燃油数量
            fuel_supplied = min(jet[MAX_FUEL] - jet[FUEL], self.global_state[BLUE_BASE_FUEL, position[0], position[1]])
            # 更新战斗机燃油量
            jet[FUEL] += fuel_supplied
            self.global_state[JET_FUEL, jet[ROW], jet[COL]] += fuel_supplied
            self.global_state[BLUE_BASE_FUEL, position[0], position[1]] -= fuel_supplied
            reward += fuel_supplied/10
        # 补给导弹
        elif supply_type == ACT_SUPPLY_MISSILES:
            # 计算补给导弹数量
            missile_supplied = min(jet[MAX_MISSILE] - jet[MISSILE], self.global_state[BLUE_BASE_MISSILE, position[0], position[1]])
            # 更新战斗机导弹量
            jet[MISSILE] += missile_supplied
            self.global_state[JET_MISSILE, jet[ROW], jet[COL]] += missile_supplied
            self.global_state[BLUE_BASE_MISSILE, position[0], position[1]] -= missile_supplied
            reward += missile_supplied/10
        return reward

    def execute_action(self, jet_index, action):
        if action == ACT_MOVE_UP:
            new_pos = (self.jets[jet_index][ROW] - 1, self.jets[jet_index][COL])
            return self.move(jet_index, new_pos)
        elif action == ACT_MOVE_DOWN:
            new_pos = (self.jets[jet_index][ROW] + 1, self.jets[jet_index][COL])
            return self.move(jet_index, new_pos)
        elif action == ACT_MOVE_LEFT:
            new_pos = (self.jets[jet_index][ROW], self.jets[jet_index][COL] - 1)
            return self.move(jet_index, new_pos)
        elif action == ACT_MOVE_RIGHT:
            new_pos = (self.jets[jet_index][ROW], self.jets[jet_index][COL] + 1)
            return self.move(jet_index, new_pos)
        elif action == ACT_ATTACK_UP:
            return self.attack(jet_index, (self.jets[jet_index][ROW] - 1, self.jets[jet_index][COL]))
        elif action == ACT_ATTACK_DOWN:
            return self.attack(jet_index, (self.jets[jet_index][ROW] + 1, self.jets[jet_index][COL]))
        elif action == ACT_ATTACK_LEFT:
            return self.attack(jet_index, (self.jets[jet_index][ROW], self.jets[jet_index][COL] - 1))
        elif action == ACT_ATTACK_RIGHT:
            return self.attack(jet_index, (self.jets[jet_index][ROW], self.jets[jet_index][COL] + 1))
        elif action == ACT_SUPPLY_FUEL:
            return self.supply(jet_index, (self.jets[jet_index][ROW], self.jets[jet_index][COL]), ACT_SUPPLY_FUEL)
        elif action == ACT_SUPPLY_MISSILES:
            return self.supply(jet_index, (self.jets[jet_index][ROW], self.jets[jet_index][COL]), ACT_SUPPLY_MISSILES)
        elif action == ACT_NO_OP:
            self.illegal_moves += 1
            self.no_op_steps += 1
            return -100



    def step(self, action):
        assert not self.terminated, "游戏已经结束，无法执行动作"
        assert not self.truncated, "游戏已经截断，无法执行动作"
        # 使用断言检查action_mask对应部分是否为1
        # assert not (action != ACT_NO_OP and self.create_action_mask()[action] == 0), "非法动作"

        reward = 0
        # 获取当前战斗机信息
        jet = self.jets[self.current_jet]
        jet_index = self.current_jet

        # 执行动作
        reward +=self.execute_action(jet_index, action)

        # 更新局部观察
        local_obs = self.extract_local_observation(
            self.global_state,
            jet[0:2],
            self.local_obs_window,
            8)

        self.turn_step += 1
        self.total_step += 1

        # 判断游戏是否结束
        self.terminated = self.check_termination()
        if self.terminated:
            print(f"Turn {self.turn} Total step:{self.total_step} Red Base destroyed:{self.destroyed_red_bases} No op steps:{self.no_op_steps}\n"
                  f"Legal:Illegal = {self.total_step}:{self.illegal_moves} \n"
                  f"Win! all red bases have been destroyed")
            self.render()
            reward += 1000

        # 判断游戏是否截断
        self.truncated = self.check_truncation()
        if self.truncated:
            print(f"Turn {self.turn} Total step:{self.total_step} Red Base destroyed:{self.destroyed_red_bases} No op steps:{self.no_op_steps} \n"
                  f"Legal:Illegal = {self.total_step}:{self.illegal_moves} \n"
                  f"Lose! max turns reached or all jets have no fuel and not in blue base")
            self.render()
            reward -= 1000


        # 在战斗机执行移动动作后，切换到下一个战斗机
        if action == ACT_NO_OP:
            assert not jet[CAN_MOVE], "战斗机没有合法动作，但是CAN_MOVE标志为True"
            self.current_jet = (self.current_jet + 1) % len(self.jets)

        # 更新回合数
        if all(jet[CAN_MOVE] == False for jet in self.jets) or self.turn_step >= 20:
            # print("All jets have moved, next turn")
            self.turn += 1
            self.turn_step = 0
            # 使得所有战斗机都能移动
            for jet in self.jets:
                if jet[FUEL] > 0 and jet[CAN_MOVE] == False:
                    jet[CAN_MOVE] = True
                    # 在地图上标记所有战斗机都能移动
                    self.global_state[JET_NOT_MOVED, jet[0], jet[1]] += 1


        return {
            "local_obs": local_obs,
            "global_obs": self.global_state,
            'action_mask': self.create_action_mask()
        }, reward, self.terminated, self.truncated,{}

    def render(self):
        print(f"Jets' position: {[(jet[ROW], jet[COL]) for jet in self.jets]}")
        for row in range(self.map_size[0]):
            for col in range(self.map_size[1]):
                if self.global_state[JET_POSITION,row,col] > 0:
                    print(f"{self.global_state[JET_POSITION,row,col]}", end=" ")
                elif self.global_state[RED_BASE_DEFENSE,row,col] > 0:
                    print("R", end=" ")
                elif self.global_state[BLUE_BASE_FUEL,row,col] > 0 or self.global_state[BLUE_BASE_MISSILE,row,col] > 0:
                    print("B", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()

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
        row, col = map(int, lines[index].strip().split())
        index += 1
        fuel, missile, defense, value = map(int, lines[index].strip().split())
        blue_bases.append((row, col, fuel, missile, defense, value))
        index += 1

    # 读取红方基地信息
    red_bases = []
    red_bases_count = int(lines[index].strip())
    index += 1
    for _ in range(red_bases_count):
        row, col = map(int, lines[index].strip().split())
        index += 1
        fuel, missile, defense, value = map(int, lines[index].strip().split())
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

if __name__ == "__main__":
    # 读取测试用例
    map_size, blue_bases, red_bases, jets = read_data_file("testcase/test1.txt")
    # 创建环境
    env = CombatEnv(map_size, blue_bases, red_bases, jets)
    # 重置环境
    env.reset()
    # 执行动作
    action = ACT_SUPPLY_FUEL
    obs, reward, terminated, truncated,info = env.step(action)
    # 打印观察
    print(obs)
    # 打印奖励
    print(f'reward:{reward}')
    # 打印游戏是否结束
    print(terminated)
    # 打印游戏是否截断
    print(truncated)
    # 渲染游戏
    env.render()
    # 打印全局观察
    print(obs["global_obs"])
    # 打印局部观察
    print(obs["local_obs"])
    # 打印回合数
    print(env.turn)
    # 打印当前战斗机索引
    print(env.current_jet)
    # 打印是否游戏结束
    print(env.terminated)
    # 打印是否游戏截断
    print(env.truncated)
