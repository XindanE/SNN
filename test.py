import snntorch as snn
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import os

# 数据加载和预处理
batch_size = 128
data_path = './data/mnist'  # 确保数据路径正确

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义图像变换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# 网络架构参数
num_inputs = 28 * 28
num_hidden = 64
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
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

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

        loss_val = torch.zeros((1), dtype=torch.float, device=device)
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

                test_loss = torch.zeros((1), dtype=torch.float, device=device)
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

# 保存模型权重
weights1 = net.fc1.weight.data.cpu().numpy()
weights2 = net.fc2.weight.data.cpu().numpy()

np.savetxt("weights1.csv", weights1, delimiter=",")
np.savetxt("weights2.csv", weights2, delimiter=",")

# 读取并转换权重文件，适用于C++头文件
def convert_weights_to_cpp(weights, name):
    with open(f"{name}.hpp", "w") as f:
        f.write(f"const float {name}[{weights.shape[0]}][{weights.shape[1]}] = {{\n")
        for i in range(weights.shape[0]):
            f.write("{")
            f.write(",".join(map(str, weights[i])))
            f.write("},\n")
        f.write("};\n")

convert_weights_to_cpp(weights1, "weights1")
convert_weights_to_cpp(weights2, "weights2")

