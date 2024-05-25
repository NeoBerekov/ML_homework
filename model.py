
import numpy as np

import torch
from torch import nn
from tianshou.data import Batch


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

# 直接运行这个文件可以测试
class NetForRainbow(nn.Module):
    def __init__(self,
                 map_depth,map_size,
                 state = None,
                 action_dim = 11,
                 num_atoms = 51,
                 device = "cpu"):
        """
            Initialize the neural network.

            Args:
                map_depth (int): The depth of the map, which corresponds to the number of channels in the input tensor.
                map_size (tuple): A tuple representing the size of the map (height, width).
                state (optional): An optional argument representing the state of the agent.
                action_dim (int, optional): The dimension of the action space. Defaults to 11.
                num_atoms (int, optional): The number of atoms for the distributional Q-value. Defaults to 51.
                device (str, optional): The device to run the model on. Defaults to "cpu".
            """

        super().__init__()
        self.map_size = map_size
        self.map_depth = map_depth
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.device = device
        if state is not None:
            self.state = state
        self.num_atoms = num_atoms
        self.conv = nn.Sequential(
            nn.Conv2d(map_depth, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(start_dim=1),
        )
        self.position_embedding = nn.Embedding(self.map_size[0]*self.map_size[1], 8)
        self.fuel_embedding = nn.Embedding(5000, 8)
        self.missile_embedding = nn.Embedding(5000, 8)
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        self.decision_layer = nn.Sequential(
            nn.Linear(576 + 8*3, 512), # 更新维度以适应卷积输出和嵌入的结合
            nn.ReLU(),
            nn.Linear(512, action_dim * num_atoms),
        )

    def forward(self, obs,state=None,info=None):
        """
        Forward pass of the neural network.

        Args:
            obs (dict): A dictionary containing the observation from the environment.
                        It should have the following keys:
                        - "observation": a numpy array or torch tensor representing the environment state.
                        - "position": a numpy array or torch tensor representing the agent's position.
                        - "fuel": a numpy array or torch tensor representing the agent's fuel level.
                        - "missile": a numpy array or torch tensor representing the agent's missile count.
            state (optional): An optional argument representing the state of the agent.
            info (optional): An optional argument representing additional information about the agent.

        Returns:
            logits (torch.Tensor): The output of the decision layer of the network.
                                   It represents the distribution over actions for the agent.
            state: The state of the agent.(unused)
            info: Additional information about the agent.(unused)
        """
        observation = obs["map_obs"]
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
        # todo: 增加一个通道，软编码地图的不可移动区块（如地方基地、边界外等）

        observation = observation.to(self.device)
        position = position.to(self.device)
        fuel = fuel.to(self.device)
        missile = missile.to(self.device)

        map_obs = self.conv(observation)
        position_idx = position[:, 0] * self.map_size[1] + position[:, 1]
        pos_emb = self.position_embedding(position_idx).view(-1, 8)
        fuel_emb = self.fuel_embedding(fuel.long().squeeze()).view(-1, 8)
        missile_emb = self.missile_embedding(missile.long().squeeze()).view(-1, 8)

        # Concatenate all features
        combined = torch.cat([map_obs, pos_emb, fuel_emb, missile_emb], dim=1)
        logits = self.decision_layer(combined).view(-1, self.action_dim, self.num_atoms)
        return logits.to("cpu"),state

        # map_obs = self.conv(observation)
        # batch_size = map_obs.size(0)
        # map_obs = map_obs.view(batch_size, 64, 3 * 3).permute(2, 0, 1)  # 变换为(seq_length, batch_size, embed_dim)
        # # 应用自注意力机制
        # map_obs, _ = self.attention(map_obs, map_obs, map_obs)
        # map_obs = map_obs.permute(1, 0, 2).contiguous()  # 恢复(batch_size, seq_length, embed_dim)
        # map_obs = map_obs.view(batch_size, -1)  # 展平为(batch_size, embed_dim*seq_length)
        #
        # # 获取位置、燃油和导弹嵌入，并确保维度匹配
        # pos_emb = self.position_embedding(position).view(batch_size, -1)
        # fuel_emb = self.fuel_embedding(fuel.long().squeeze()).view(batch_size, -1)  # 确保燃油数值是整数并进行嵌入
        # missile_emb = self.missile_embedding(missile.long().squeeze()).view(batch_size, -1)  # 确保导弹数值是整数并进行嵌入
        #
        # # 结合所有特征
        # combined = torch.cat([map_obs, pos_emb, fuel_emb, missile_emb], dim=1)
        # logits = self.decision_layer(combined).view(batch_size, self.num_atoms, -1)
        # logits = logits.permute(0, 2, 1)  # 调整维度为(batch_size, action_dim, num_atoms)

if __name__ == "__main__":
    # 示例使用
    map_depth = 8
    map_size = (6, 9)  # Height = 6, Width = 9
    action_dim = 11
    num_atoms = 51

    model = NetForRainbow(map_depth, map_size, action_dim=action_dim, num_atoms=num_atoms)

    # 设置批量大小为10
    batch_size = 10

    # 生成示例输入
    observation = torch.randn(batch_size, map_depth, map_size[0], map_size[1])  # Batch size = 10
    # Generate position as (x, y) coordinates for each batch item
    position_x = torch.randint(0, map_size[0], (batch_size,))  # Random x coordinates for batch
    position_y = torch.randint(0, map_size[1], (batch_size,))  # Random y coordinates for batch
    position = torch.stack((position_x, position_y), dim=1)  # Combine into a single tensor

    fuel = torch.randint(0, 5000, (batch_size,))  # Fuel levels for each batch item
    missile = torch.randint(0, 5000, (batch_size,))  # Missile counts for each batch item
    obs = {"map_obs": observation, "position": position, "fuel": fuel, "missile": missile}

    # 通过网络前向传播
    logits, _, __ = model.forward(obs)
    print(logits.shape)  # Expected to output [10, 11, 51], representing the action distribution for each batch item