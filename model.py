
import numpy as np

import torch
from torch import nn


#  self.observation_spaces = {
#             agent: spaces.Dict({
#                 "observation": spaces.Box(low=-0.001, high=5000.0, shape=(8,map_size[0],map_size[1]), dtype=np.float32),
#
#                 "action_mask": spaces.MultiBinary(11),
#
#                 "position": spaces.Tuple((spaces.Discrete(map_size[0]), spaces.Discrete(map_size[1]))),
#
#                 "fuel": spaces.Box(low=-0.001, high=5000, shape=(1,), dtype=np.float32),
#
#                 "missile": spaces.Box(low=-0.001, high=5000, shape=(1,), dtype=np.float32)
#             }) for agent in self.agents
class NetForRainbow(nn.Module):
    def __init__(self,
                 map_depth,map_size,
                 state = None,
                 action_dim = 11,
                 num_atoms = 51,
                 device = "cpu"):
        super().__init__()
        if state is not None:
            self.state = state
        self.device = device
        self.num_atoms = num_atoms
        self.conv = nn.Sequential(
            nn.Conv2d(map_depth, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(start_dim=1),
        )
        self.position_embedding = nn.Embedding(map_size[0]*map_size[1], 8)
        self.fuel_embedding = nn.Embedding(5000, 8)
        self.missile_embedding = nn.Embedding(5000, 8)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.decision_layer = nn.Sequential(
            nn.Linear(64 * 9 + 8 * 3, 512),  # 更新维度以适应卷积输出和嵌入的结合
            nn.ReLU(),
            nn.Linear(512, action_dim * num_atoms),
        )

    def forward(self, obs,state=None,info=None):
        observation = obs["observation"]
        position = obs["position"]
        fuel = obs["fuel"]
        missile = obs["missile"]
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32)
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=torch.long)
        if not isinstance(fuel, torch.Tensor):
            fuel = torch.tensor(fuel, dtype=torch.float32)
        if not isinstance(missile, torch.Tensor):
            missile = torch.tensor(missile, dtype=torch.float32)
        # 卷积层处理观测地图
        map_obs = self.conv(observation)
        batch_size = map_obs.size(0)
        map_obs = map_obs.view(batch_size, 64, 3 * 3).permute(2, 0, 1)  # 变换为(seq_length, batch_size, embed_dim)
        # 应用自注意力机制
        map_obs, _ = self.attention(map_obs, map_obs, map_obs)
        map_obs = map_obs.permute(1, 0, 2).contiguous()  # 恢复(batch_size, seq_length, embed_dim)
        map_obs = map_obs.view(batch_size, -1)  # 展平为(batch_size, embed_dim*seq_length)

        # 获取位置、燃油和导弹嵌入，并确保维度匹配
        pos_emb = self.position_embedding(position).view(batch_size, -1)
        fuel_emb = self.fuel_embedding(fuel.long().squeeze()).view(batch_size, -1)  # 确保燃油数值是整数并进行嵌入
        missile_emb = self.missile_embedding(missile.long().squeeze()).view(batch_size, -1)  # 确保导弹数值是整数并进行嵌入

        # 结合所有特征
        combined = torch.cat([map_obs, pos_emb, fuel_emb, missile_emb], dim=1)
        logits = self.decision_layer(combined).view(batch_size, self.num_atoms, -1)
        logits = logits.permute(0, 2, 1)  # 调整维度为(batch_size, action_dim, num_atoms)
        return logits,state,info


# 运行该文件可以测试
if __name__ == "__main__":
    # 示例使用
    map_depth = 8
    map_size = (10, 10)
    state_dim = (map_depth, *map_size)
    action_dim = 11
    num_atoms = 51

    model = NetForRainbow(map_depth, map_size, action_dim=action_dim, num_atoms=num_atoms)

    # 生成示例输入
    observation = torch.randn(1, map_depth, map_size[0], map_size[1])  # 假设 batch size 为 1
    position = torch.randint(0, map_size[0] * map_size[1], (1,))
    fuel = torch.randint(0, 5000, (1,))
    missile = torch.randint(0, 5000, (1,))

    # 通过网络前向传播
    logits = model(observation, position, fuel, missile)
    print(logits.shape)  # 应该输出 [1, 11, 51], 表示一个批次中的动作分布