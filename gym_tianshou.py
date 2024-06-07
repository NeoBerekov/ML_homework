import gymnasium as gym
import tianshou as ts

from gymnasium.envs.registration import register
from gymnasium_env import CombatEnv, read_data_file
from model_new import DQN

register(
    id='Combat-v0',
    entry_point='gymnasium_env:CombatEnv',
    order_enforce=True,
)
map_size, blue_bases, red_bases, jets = read_data_file("testcase/test1.txt")
map_depth = 9
num_actions = 11
local_obs_window = 11
boundary_gradient = 10
max_turns = 50


# 初始化环境
def get_env():
    return gym.make('Combat-v0', map_size=map_size, blue_bases=blue_bases, red_bases=red_bases, jets=jets,
                    local_obs_window=local_obs_window,
                    boundary_gradient=boundary_gradient,
                    max_turns=max_turns,
                    )

train_envs = ts.env.DummyVectorEnv([get_env for _ in range(4)])
test_envs = ts.env.DummyVectorEnv([get_env for _ in range(4)])

