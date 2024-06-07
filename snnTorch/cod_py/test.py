import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch.onnx

# 数据加载和预处理
batch_size = 128
data_path = '/data/mnist'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(torch.cuda.is_available())


# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# 网络架构参数
num_inputs = 28 * 28
num_hidden = 1000
num_outputs = 10

num_steps = 25
beta = 0.95


# 定义神经网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # 在正确的设备上初始化隐藏状态
        mem1 = self.lif1.init_leaky().to(device)
        mem2 = self.lif2.init_leaky().to(device)

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


net = Net().to(device)

# 训练和测试
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 1
loss_hist = []
test_loss_hist = []
train_accuracy_hist = []
test_accuracy_hist = []

for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val += loss(mem_rec[step], targets)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        loss_hist.append(loss_val.item())

        if iter_counter % 50 == 0:
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                test_spk, test_mem = net(test_data.view(batch_size, -1))

                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                _, idx = test_spk.sum(dim=0).max(1)
                test_acc = np.mean((test_targets == idx).detach().cpu().numpy())
                test_accuracy_hist.append(test_acc)

                output, _ = net(data.view(batch_size, -1))
                _, idx = output.sum(dim=0).max(1)
                train_acc = np.mean((targets == idx).detach().cpu().numpy())
                train_accuracy_hist.append(train_acc)

                print(f"Epoch {epoch}, Iteration {iter_counter}")
                print(f"Train Set Loss: {loss_hist[-1]:.2f}, Accuracy: {train_acc * 100:.2f}%")
                print(f"Test Set Loss: {test_loss_hist[-1]:.2f}, Accuracy: {test_acc * 100:.2f}%")
                print("\n")

        iter_counter += 1


# 保存模型
torch.save(net.state_dict(), "model.pth")

# # 设置量化配置
# net.qconfig = torch.quantization.default_qconfig
#
# # 准备模型进行量化
# torch.quantization.prepare(net, inplace=True)
#
# # 使用一部分训练数据进行校准
# with torch.no_grad():
#     for data, _ in train_loader:
#         net(data.to(device))
#
# # 转换模型
# torch.quantization.convert(net, inplace=True)
#
# # 保存量化模型
# torch.save(net.state_dict(), "quantized_model.pth")







fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(train_accuracy_hist, label="Train Accuracy")
plt.plot(test_accuracy_hist, label="Test Accuracy")
plt.title("Accuracy Curves")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()

# for name, param in net.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
#         torch.save(param.data, f'{name}.pt')



# 在测试集上测试
total = 0
correct = 0

test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        test_spk, _ = net(data.view(data.size(0), -1))

        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")