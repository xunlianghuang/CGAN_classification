from PIL import Image
import numpy as np

# # 生成一个随机的灰度图像
# test_image = np.random.randint(0, 256, size=(28, 28,3), dtype=np.uint8)
#
# # 将NumPy数组转换为PIL图像
# pil_image = Image.fromarray(test_image).convert("L")

# 保存图像
# pil_image.save('random_grayscale_image.png')
im  =  Image.open("data/test/1/0.jpg")  #读入图片数据
img =  np.array(im)  #转换为numpy
print(1)