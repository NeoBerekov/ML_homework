import os
from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy,RainbowPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net

from pettingzoo.utils import TerminateIllegalWrapper


from pettingzoo_env import read_data_file, CustomMilitaryEnv
from model import NetForRainbow

map_size,blue_bases,red_bases,jets = read_data_file("testcase/test1.txt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return PettingZooEnv(
        TerminateIllegalWrapper(
            CustomMilitaryEnv(map_size,blue_bases,red_bases,jets),
            illegal_reward=-10000
        )
    )

# class NetForRainbow(nn.Module):
#     def __init__(self,
#                  map_depth,map_size,
#                  state = None,
#                  action_dim = 11,
#                  num_atoms = 51,
#                  device = "cpu"):
def _get_agents(num_agents):
    env = _get_env()
    observation_space = env.observation_space
    map_observation_space = env.observation_space["observation"]
    position_space = env.observation_space["position"]
    fuel_space = env.observation_space["fuel"]
    missile_space = env.observation_space["missile"]

    action_space = env.action_space.shape or env.action_space.n

    net = NetForRainbow(
        map_depth=map_observation_space.shape[0],
        map_size=map_observation_space.shape[1:],
        device=device
    ).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    shared_policy = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor=0.9,
        estimation_step=3,
        target_update_freq=320,
        v_min=-100.0,
        v_max=5000.0,
        num_atoms=51
    )
    policies = [shared_policy for _ in range(num_agents)]
    multiagent_policy = MultiAgentPolicyManager(policies, env)
    return multiagent_policy, optim, env.agents

# if __name__ == "__main__":
#     num_agents = len(jets)  # Set this to the number of cooperating agents
#     train_envs = DummyVectorEnv([_get_env for _ in range(10)])
#     test_envs = DummyVectorEnv([_get_env for _ in range(10)])
#     np.random.seed(114514)
#     train_envs.seed(114514)
#     test_envs.seed(114514)
#
#     policy, optim, agents = _get_agents(num_agents)
#
#     train_collector = Collector(
#         policy,
#         train_envs,
#         VectorReplayBuffer(20000, len(train_envs)),
#         exploration_noise=True
#     )
#     test_collector = Collector(
#         policy,
#         test_envs,
#         exploration_noise=True
#     )
#     train_collector.collect(n_step=64 * 10)
#
#
#     def save_best_fn(policy):
#         model_save_path = os.path.join("log", "military", "rainbow", "policy.pth")
#         os.makedirs(os.path.join("log", "military", "rainbow"), exist_ok=True)
#         torch.save(policy.policies[agents[0]].state_dict(), model_save_path)
#
#     def save_checkpoint_fn(epoch, env_step, gradient_step):
#         model_save_path = os.path.join("log", "military", "rainbow", f"policy_{env_step}.pth")
#         os.makedirs(os.path.join("log", "military", "rainbow"), exist_ok=True)
#         torch.save(policy.policies[agents[0]].state_dict(), model_save_path)
#         return model_save_path
#
#
#     def stop_fn(mean_rewards):
#         return mean_rewards >= 10000  # 停止训练的条件，可根据需要调整
#
#
#     def train_fn(epoch, env_step):
#         policy.policies[agents[0]].set_eps(0.1)  # 设置 epsilon-greedy 策略中的 epsilon
#
#
#     def test_fn(epoch, env_step):
#         policy.policies[agents[0]].set_eps(0.05)  # 测试时减少探索
#
#
#     result = OffpolicyTrainer(
#         policy=policy,
#         train_collector=train_collector,
#         test_collector=test_collector,
#         max_epoch=500,
#         step_per_epoch=20000,
#         step_per_collect=50,
#         episode_per_test=10,
#         batch_size=64,
#         update_per_step=0.1,
#         test_in_train=False,
#         reward_metric=lambda rews: np.sum(rews, axis=1),  # Sum of rewards across all agents
#         save_best_fn=save_best_fn,
#         stop_fn=stop_fn,
#         train_fn=train_fn,
#         test_fn=test_fn
#     ).run()
#
#     print(f"\n==========Result==========\n{result}")
