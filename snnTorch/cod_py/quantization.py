import snntorch as snn
from snntorch.functional import quant
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# 设置参数
batch_size = 128
data_path = '/data/mnist'
num_inputs = 28 * 28
num_hidden = 1000  # 确保与 QuantizedSNN 一致
num_outputs = 10
num_steps = 25
beta = 0.95
threshold = 1
num_bits = 8
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


class QuantizedSNN(nn.Module):
    def __init__(self):
        super(QuantizedSNN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.q_lif1 = quant.state_quant(num_bits=num_bits, uniform=True, threshold=threshold)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, state_quant=self.q_lif1)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.q_lif2 = quant.state_quant(num_bits=num_bits, uniform=True, threshold=threshold)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, state_quant=self.q_lif2)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
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


net = QuantizedSNN().to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))  # 调整学习率

num_epochs = 1  # 减少 epoch 数
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

        if iter_counter % 10 == 0:  # 调整评估频率
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

                print("Epoch {}, Iteration {}".format(epoch, iter_counter))
                print("Train Set Loss: {:.2f}, Accuracy: {:.2f}%".format(loss_hist[-1], train_acc * 100))
                print("Test Set Loss: {:.2f}, Accuracy: {:.2f}%".format(test_loss_hist[-1], test_acc * 100))
                print("\n")

        iter_counter += 1

torch.save(net.state_dict(), './model_test.pth')
