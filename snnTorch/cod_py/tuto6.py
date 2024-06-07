import snntorch as snn
from snntorch import surrogate
# from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(torch.cuda.is_available())

# Leaky neuron model, overriding the backward pass with a custom function
class LeakySigmoidSurrogate(nn.Module):
    def __init__(self, beta, threshold=1.0, k=25):
        super(LeakySigmoidSurrogate, self).__init__()
        self.beta = beta
        # self.threshold = torch.tensor([threshold], device=device)  # 确保threshold也在正确的设备上
        self.k = k
        self.device = device
        # Leaky_Surrogate is defined in the previous tutorial and not used here
        # super(Leaky_Surrogate, self).__init__()

        # initialize decay rate beta and threshold
        # self.beta = beta
        self.threshold = threshold
        self.surrogate_func = self.FastSigmoid.apply

    # the forward function is called each time we call Leaky
    def forward(self, input_, mem):
        input_ = input_.to(self.device)
        mem = mem.to(self.device)
        spk = self.surrogate_func((mem - self.threshold))  # call the Heaviside function
        reset = (spk - self.threshold).detach()
        mem = self.beta * mem + input_ - reset
        return spk, mem

    # Forward pass: Heaviside function
    # Backward pass: Override Dirac Delta with gradient of fast sigmoid
    @staticmethod
    class FastSigmoid(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem, k=25):
            ctx.save_for_backward(mem)  # store the membrane potential for use in the backward pass
            ctx.k = k
            out = (mem > 0).float()  # Heaviside on the forward pass: Eq(1)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (mem,) = ctx.saved_tensors  # retrieve membrane potential
            grad_input = grad_output.clone()
            grad = grad_input / (ctx.k * torch.abs(mem) + 1.0) ** 2  # gradient of fast sigmoid on backward pass: Eq(4)
            return grad, None


spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5

lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

# dataloader arguments
batch_size = 128
data_path = '/data/mnist'

dtype = torch.float


# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = nn.Linear(64*4*4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

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


#  Initialize Network
net = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*4*4, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

# forward pass
data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

# for step in range(num_steps):
#     spk_out, mem_out = net(data)


def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


spk_rec, mem_rec = forward_pass(net, num_steps, data)

# already imported snntorch.functional as SF
loss_fn = SF.ce_rate_loss()
loss_val = loss_fn(spk_rec, targets)

print(f"The loss from an untrained network is {loss_val.item():.3f}")
