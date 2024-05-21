from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy

from pettingzoo.utils import TerminateIllegalWrapper

from pettingzoo_env import CustomMilitaryEnv, read_data_file

map_size, blue_bases, red_bases, jets = read_data_file("testcase/test1.txt")
env = CustomMilitaryEnv(map_size, blue_bases, red_bases, jets)

env = TerminateIllegalWrapper(env,illegal_reward=-10000)

env.reset()

env = PettingZooEnv(env)

policies = MultiAgentPolicyManager([RandomPolicy(env.action_space) for _ in range(len(env.agents))],env)

env = DummyVectorEnv([lambda: env])

collector = Collector(policies,env)

result = collector.collect(n_episode=1, render=1)

