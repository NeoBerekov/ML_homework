import numpy as np


def extract_local_observation(full_map, position, window_size, gradient_depth):
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

    return local_obs


# 示例用法
# 假设有一个地图和一个位置
map_size = (10, 10)
full_map = np.random.rand(8, *map_size)  # 假设地图有8个深度
position = (0, 0)  # 智能体的位置
window_size = 5  # 5*5的观察窗口
gradient_depth = 7  # 假设第8层是梯度层

local_obs = extract_local_observation(full_map, position, window_size, gradient_depth)
print(local_obs)