from gymnasium_env import CombatEnv,read_data_file
from model_new import DQN
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import random
from collections import deque



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


# 重放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, map_size, map_depth, num_actions,local_obs_window,replay_buffer_capacity=100000, batch_size=1024, lr=1e-4, gamma=0.99):
        self.map_size = map_size
        self.map_depth = map_depth
        self.num_actions = num_actions
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(map_size=map_size, map_depth=map_depth,local_obs_window=local_obs_window, num_actions=num_actions).to(self.device)
        self.target_model = DQN(map_size=map_size, map_depth=map_depth,local_obs_window=local_obs_window, num_actions=num_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.steps = 0

    def check_gradients(self):
        for param in self.model.parameters():
            if param.grad is not None:
                assert param.grad.data.norm(2).item() != 0, "Gradient vanishing detected!"

    def act(self, state, epsilon):
        if random.random() < epsilon:
            valid_actions = state['action_mask']
            valid_action_indices = np.where(valid_actions == 1)[0]
            if valid_action_indices.size == 0:
                choice = ACT_NO_OP
            else:
                choice = np.random.choice(valid_action_indices)
            assert valid_actions[choice] == 1 or choice == ACT_NO_OP, f"Invalid action: {choice}"
            return choice
        else:
            local_state = state['local_obs']
            local_state = torch.FloatTensor(local_state).to(self.device)
            local_state = local_state.unsqueeze(0)
            q_values = self.model(local_state)

            action_mask = state['action_mask']
            action_mask = torch.FloatTensor(action_mask).to(self.device)
            invalid_action_mask = (1 - action_mask) * -1e9
            masked_q_values = q_values + invalid_action_mask

            return masked_q_values.argmax().item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self):
        if len(self.replay_buffer) < self.batch_size * 4:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = np.array([state['local_obs'] for state in states])
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = np.array([state['local_obs'] for state in next_states])
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1).values
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)
        loss = self.loss_fn(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        # self.check_gradients()
        self.optimizer.step()
        self.steps += 1

        if self.steps % 4000 == 0:
            self.update_target_model()


map_size,blue_bases,red_bases,jets = read_data_file("testcase/test1.txt")
map_depth = 9
num_actions = 11
local_obs_window = 11
boundary_gradient = 10
max_turns = 50
# 初始化环境
env = CombatEnv(map_size, blue_bases, red_bases, jets,
                local_obs_window=local_obs_window,
                boundary_gradient=boundary_gradient,
                max_turns=max_turns,
                )
agent = DQNAgent(map_size=map_size, map_depth=map_depth, num_actions=num_actions,local_obs_window=local_obs_window)

# 训练
num_episodes = 200
state = env.reset()
best_reward = -np.inf
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    epsilon = 1
    count = 0
    while not done:
        epsilon = max(0.1, 0.9 - episode / 120)
        action = agent.act(state, epsilon)
        next_state, reward, terminated,truncated,_ = env.step(action)
        done = terminated or truncated
        # if action != ACT_NO_OP and count % 10 == 0:
        #     agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        agent.train()
        count += 1
    if total_reward > best_reward:
        best_reward = total_reward
    print(f"Episode {episode}, total reward: {total_reward}, epsilon:{epsilon}, \nbuffer size: {len(agent.replay_buffer)} \n")
print(f"Best reward: {best_reward}")