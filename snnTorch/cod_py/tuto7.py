import tonic
import tonic.transforms as transforms
import timeit
import numpy as np  # 用于计算平均值和标准差
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
# torch.cuda.empty_cache()
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'



dataset = tonic.datasets.NMNIST(save_to='./data', train=True)
events, target = dataset[0]

print(events)
tonic.utils.plot_event_grid(events)

sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000), transforms.ToFrame(sensor_size=sensor_size,
                                    time_window=1000)])

trainset = tonic.datasets.NMNIST(save_to='./tmp/data', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./tmp/data', transform=frame_transform, train=False)


# Fast DataLoading
from torch.utils.data import DataLoader
from tonic import DiskCachedDataset

#
# cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
# cached_dataloader = DataLoader(cached_trainset)
#
# batch_size = 128
# trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
#
#
# def load_sample_batched():
#     events, target = next(iter(cached_dataloader))


# # 执行重复的时间测量
# # repeat=10 指进行 10 次测试，每次测试执行 100 次函数调用
# results = timeit.repeat('load_sample_batched()', 'from __main__ import load_sample_batched', number=100, repeat=10)
#
# # 计算平均时间和标准差，将时间转换为毫秒
# mean_time = np.mean(results) * 1000
# std_dev = np.std(results) * 1000
#
# # 输出结果，格式化为与 %timeit 类似的输出
# print(f"{mean_time:.1f} ms ± {std_dev:.0f} µs per loop (mean ± std. dev. of 10 runs, 100 loops each)")

# speed up dataloading further by caching to main memory instead of to disk
# from tonic import MemoryCachedDataset
# cached_trainset = MemoryCachedDataset(trainset)

# Training our network using frames created from events
import torch
import torchvision

transform = tonic.transforms.Compose([torch.from_numpy, torchvision.transforms.RandomRotation([-10,10])])

cached_trainset = DiskCachedDataset(trainset, transform=transform, cache_path='./cache/nmnist/train')
# no augmentations for the testset
cached_testset = DiskCachedDataset(testset, cache_path='./cache/nmnist/test')

batch_size = 32
trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
testloader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

event_tensor, target = next(iter(trainloader))
print(event_tensor.shape)

# Defining our network
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn as nn

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# neuron and simulation parameters
spike_grad = surrogate.atan()
beta = 0.5


# #  Initialize Network
# net = nn.Sequential(nn.Conv2d(2, 12, 5),
#                     nn.MaxPool2d(2),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#                     nn.Conv2d(12, 32, 5),
#                     nn.MaxPool2d(2),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
#                     nn.Flatten(),
#                     nn.Linear(32*5*5, 10),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
#                     ).to(device)
# print(f"Net is on device: {next(net.parameters()).device}")
# # net = net.to(device)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(2, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 32, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(32 * 5 * 5, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=True)

    def forward(self, x):

        # Initialize hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky().to(device)
        mem2 = self.lif2.init_leaky().to(device)
        mem3 = self.lif3.init_leaky().to(device)

        cur1 = F.max_pool2d(self.conv1(x), 2)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = F.max_pool2d(self.conv2(spk1), 2)
        spk2, mem2 = self.lif2(cur2, mem2)

        cur3 = self.fc1(spk2.view(batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)
        return spk3, mem3


# 创建网络实例
net = Net()  # 确保传入设备参数


# this time, we won't return membrane as we don't need it
def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net
    net = net.to(device)
    data = data.to(device)
    for step in range(data.size(0)):  # data.size(0) = number of time steps
        # print(f"data is on device gpu:")
        # print(data.is_cuda)
        # print(f"Network is on device: {next(net.parameters()).device}")
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)
        spk_rec = spk_rec
    return torch.stack(spk_rec)


optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.9, incorrect_rate=0.1)

num_epochs = 1
num_iters = 50

loss_hist = []
acc_hist = []

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(trainloader)):
        with torch.autograd.set_detect_anomaly(True):

            data = data.to(device)
            targets = targets.to(device)

            net.train()
            net = net.to(device)
            # print(f"data is on device gpu:")
            # print(data.is_cuda)
            # print(f"Network is on device: {next(net.parameters()).device}")
            spk_rec = forward_pass(net, data).to(device)
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")

            # training loop breaks after 50 iterations
            if i == num_iters:
                break



# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(acc_hist)
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()