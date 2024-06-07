import os
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置数据路径和输出路径
data_path = "/home/xindan/snn_project/data/mnist"
output_dir_steps = "/home/xindan/snn_project/snn_lif"
output_dir_images = "/home/xindan/snn_project/data/mnist_time_images"
os.makedirs(output_dir_steps, exist_ok=True)
os.makedirs(output_dir_images, exist_ok=True)

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# 随机选择样本
num_samples = 10  # 这里设置选择的样本数量
indices = np.random.choice(len(mnist_train), num_samples, replace=False)

# 将图像数据转换为时间步格式
time_steps = 30

for i, index in enumerate(indices):
    image, label = mnist_train[index]
    image_flatten = image.flatten().numpy()
    
    # 生成txt文件
    txt_filename = os.path.join(output_dir_steps, f"st{i + 1}.txt")
    with open(txt_filename, "w") as file:
        for _ in range(time_steps):
            binary_image = (np.random.rand(784) < image_flatten).astype(int)  # 使用rate coding进行二值化
            for pixel in binary_image:
                file.write(f"{pixel} ")
            file.write("\n")
    
    # 保存图像为png文件
    png_filename = os.path.join(output_dir_images, f"st{i + 1}.png")
    plt.imsave(png_filename, image.squeeze(), cmap='gray')

    # 打印文件名和对应的标签
    print(f"Generated {txt_filename} and {png_filename} for label: {label}")

    # 为了验证每次的随机选择，可以打印每个样本的标签
    print(f"Random sample {i + 1}: Label {label}")

