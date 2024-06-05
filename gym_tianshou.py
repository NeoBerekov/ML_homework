import os

import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.gym_wrappers import
from tianshou.policy import RainbowPolicy, MultiAgentPolicyManager,C51Policy
from tianshou.trainer import offpolicy_trainer

from pettingzoo.utils import TerminateIllegalWrapper


from gymnasium_env import CombatEnv,read_data_file
from model import NetForRainbow

from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger



# 读test文件
map_size,blue_bases,red_bases,jets = read_data_file("testcase/test2.txt")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    return


def _get_agents(num_agents):
    env = _get_env()
    map_observation_space = env.observation_space["observation"]["map_obs"]

    net = NetForRainbow(
        map_depth=map_observation_space.shape[0],
        map_size=map_observation_space.shape[1:],
        device=device
    ).to(device)

    optim = torch.optim.Adam(net.parameters(), lr=1e-4)

    # # 记得看RainbowPolicy的输入输出要求
    # shared_policy = RainbowPolicy(
    #     model=net,
    #     optim=optim,
    #     action_space=env.action_space,
    #     discount_factor=0.9,
    #     estimation_step=10,
    #     target_update_freq=200,
    #     v_min=-300.0,
    #     v_max=5000.0,
    #     num_atoms=51
    # )
    policies = [RainbowPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=0.9,
        estimation_step=10,
        target_update_freq=200,
        v_min=-300.0,
        v_max=5000.0,
        num_atoms=51
    ) for _ in range(num_agents)]
    multiagent_policy = MultiAgentPolicyManager(policies, env,observation_space=env.observation_space)
    return multiagent_policy, optim, env.agents


# Logging
writer = SummaryWriter(log_dir='./logs')
logger = TensorboardLogger(writer)
print("Logging to ./logs")


if __name__ == "__main__":
    num_agents = len(jets)  # Set this to the number of cooperating agents
    train_envs = DummyVectorEnv([_get_env for _ in range(5)])
    test_envs = DummyVectorEnv([_get_env for _ in range(5)])
    np.random.seed(114514)
    train_envs.seed(114514)
    test_envs.seed(114514)

    policy, optim, agents = _get_agents(num_agents)

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20000, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(
        policy,
        test_envs,
        exploration_noise=True
    )
    train_collector.collect(n_step=32 * 10)
    train_collector.reset()


    def save_best_fn(policy):
        print("Saving best model")
        model_save_path = os.path.join("log", "military", "rainbow", "policy.pth")
        os.makedirs(os.path.join("log", "military", "rainbow"), exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        print(f"Saving model at epoch {epoch}, env step {env_step}")
        model_save_path = os.path.join("log", "military", "rainbow", f"policy_{env_step}.pth")
        os.makedirs(os.path.join("log", "military", "rainbow"), exist_ok=True)
        torch.save(policy.policies[agents[0]].state_dict(), model_save_path)
        return model_save_path


    def stop_fn(mean_rewards):
        return mean_rewards >= 1  # 停止训练的条件，没细调，可能需要调整，感觉不太够


    def train_fn(epoch, env_step):
        print(f"Train Epoch {epoch}, Env step {env_step}")
        policy.policies[agents[0]].set_eps(0.1)  # 设置 epsilon-greedy 策略中的 epsilon


    def test_fn(epoch, env_step):
        print(f"Test Epoch {epoch}, Env step {env_step}")
        policy.policies[agents[0]].set_eps(0.05)  # 测试时减少探索



    result = offpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=500,
        step_per_epoch=100,
        step_per_collect=25,
        episode_per_test=10,
        batch_size=64,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=lambda rews: np.sum(rews, axis=1),  # Sum of rewards across all agents
        save_best_fn=save_best_fn,
        stop_fn=stop_fn,
        train_fn=train_fn,
        test_fn=test_fn,
        logger=logger,
    )

    print(f"\n==========Result==========\n{result}")
