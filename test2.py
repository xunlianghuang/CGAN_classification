import torch
import numpy as np
from PIL import Image

# 创建一个随机的[32, 32, 3]大小的PyTorch张量（假设这是你的数据）
tensor = torch.rand(1, 3, 32, 32)  # 假设tensor的shape为[1, 3, 32, 32]

# 将tensor的值在[0,1]之间，转换为[0,255]之间的整数
tensor *= 255
tensor = tensor.int()

# 将tensor转换为numpy数组，并调整维度
tensor_np = tensor.squeeze().permute(1, 2, 0).numpy()  # [32, 32, 3]

# 将numpy数组转换为PIL图像
image = Image.fromarray(tensor_np.astype(np.uint8))
gray_image = image.convert('L')
gray_image = gray_image.resize((28,28))

# 保存图像
image.save('gray_image.png')
