import gym
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


class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, outputs)

    def forward(self, x):
        # print("Input shape:", x.shape)  # 查看输入形状
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # 这里调整为保持输出格式一致
            # print(policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1))
            return policy_net(state.unsqueeze(0)).max(1)[1].view(1, 1)
    else:
        # 保持探索时的输出格式和利用时的一致
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)



Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

env = gym.make('CartPole-v1').unwrapped

# 设置训练参数
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_actions = env.action_space.n
# print(env.observation_space.shape)
n_states = env.observation_space.shape[0]
policy_net = DQN(n_states, n_actions).float()
target_net = DQN(n_states, n_actions).float()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0

# 时间记录
start_time = time.time()

def optimize_model():
    # 从memory中采样
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # 计算非终止状态的mask
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
    # 将非终止状态的state拼接起来
    non_final_next_states = torch.stack([s for s in batch.next_state
                                       if s is not None])
    # 将state, action, reward转换为tensor
    state_batch = torch.stack(batch.state)
    # print("State batch shape:", state_batch.shape)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # 计算当前状态的Q值
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def main_loop(num_episodes):
    for i_episode in range(num_episodes):
        episode_time = time.time()
        print("Episode:", i_episode)
        initial_state,_ = env.reset()
        state = torch.tensor(initial_state, dtype=torch.float)
        flag = True
        for t in count():
            action = select_action(state)
            next_state, reward, done, _ = env.step(action.item())[:4]
            reward = torch.tensor([reward], dtype=torch.float)
            next_state = np.array(next_state)  # 转换为单一的 numpy 数组
            next_state = torch.tensor(next_state, dtype=torch.float)
            if done:
                next_state = None
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()
            run_time= time.time() - episode_time
            run_time = int(run_time)
            if done:
                break
            if run_time > 10 and flag:
                print("This one is hopeful!")
                torch.save(policy_net.state_dict(), f'{i_episode}_{run_time}s_model.pth')
                flag = False
            if run_time > 60:
                print("Good enough!Save a copy and break!")
                torch.save(policy_net.state_dict(), f'{i_episode}_{run_time}s_model.pth')
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        total_time = time.time() - episode_time
        print("Episode Time:", total_time)
    print('Complete')
    env.close()
    torch.save(policy_net.state_dict(), 'final_model.pth')


def test():

    test_time = time.time()
    env = gym.make('CartPole-v1', render_mode='human').unwrapped
    policy_net.load_state_dict(torch.load('316_61s_model.pth'))
    policy_net.eval()
    initial_state, _ = env.reset()
    state = torch.tensor(initial_state, dtype=torch.float)
    for t in count():
        env.render()
        if random.random() < 0.15:
            action = random.choice([0,1])
        else:
            action = policy_net(state.unsqueeze(0)).max(1)[1].item()
        next_state, reward, done, _ = env.step(action)[:4]
        state = torch.tensor(next_state, dtype=torch.float)
        if done:
            break
    print("Test Time:", time.time() - test_time)
    env.close()

# main_loop(500)
test()