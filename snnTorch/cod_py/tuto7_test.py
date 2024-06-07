import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import functional as SF
import matplotlib.pyplot as plt
from snntorch import surrogate


dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
events, target = dataset[0]

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载和转换数据集
sensor_size = tonic.datasets.NMNIST.sensor_size
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), transforms.ToFrame(sensor_size=sensor_size,
                                    time_window=1000)])

trainset = tonic.datasets.NMNIST(save_to='./tmp/data', transform=frame_transform, train=True)
cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')

batch_size = 128
trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

spike_grad = surrogate.atan()
beta = 0.5


# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.network = nn.Sequential(nn.Conv2d(2, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 32, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(32*5*5, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

    def forward(self, x):
        return self.network(x)


net = Net()


# 简单的网络前向传递函数
def forward_pass(net, data):
    data = data.to(device)
    output = net(data)
    return output


# 测试网络是否在正确的设备上
print(f"Network is on device: {next(net.parameters()).device}")


# 训练网络
optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
loss_fn = SF.mse_count_loss()

for data, targets in trainloader:
    data, targets = data.to(device), targets.to(device)
    optimizer.zero_grad()
    output = forward_pass(net, data)
    loss = loss_fn(output, targets)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")

# 绘图
plt.figure(figsize=(10, 5))
plt.plot([1, 2, 3], [loss.item() for _ in range(3)])  # 示例数据
plt.title("Loss over time")
plt.xlabel("Time Step")
plt.ylabel("Loss")
plt.show()





