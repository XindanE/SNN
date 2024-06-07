import os
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from PIL import Image

# 定义转换器，将图像转换为张量
transform = transforms.Compose([transforms.ToTensor()])

# 加载MNIST数据集（假设已经下载好）
train_dataset = MNIST(root='/home/xindan/snn_project/data/mnist', train=True, transform=transform, download=False)
test_dataset = MNIST(root='/home/xindan/snn_project/data/mnist', train=False, transform=transform, download=False)

# 创建文件夹用于保存图片
os.makedirs("mnist_images/train", exist_ok=True)
os.makedirs("mnist_images/test", exist_ok=True)

def save_images(dataset, directory, num_images=100):
    for i, (image, label) in enumerate(dataset):
        if i >= num_images:
            break
        # 将张量转换为PIL Image对象
        image = transforms.ToPILImage()(image)
        
        # 创建每个标签的子文件夹
        label_dir = os.path.join(directory, str(label))
        os.makedirs(label_dir, exist_ok=True)
        
        # 保存图像为PNG文件
        image.save(os.path.join(label_dir, f"{i}.png"))

# 保存训练集和测试集图片，限制为每个集100张图片
save_images(train_dataset, "mnist_images/train", num_images=100)
save_images(test_dataset, "mnist_images/test", num_images=100)
