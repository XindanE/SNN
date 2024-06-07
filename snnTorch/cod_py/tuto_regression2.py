import snntorch as snn
from snntorch import functional as SF

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
import tqdm


beta = 0.9 # membrane potential decay rate
num_steps = 10 # 10 time steps

# RLIF Neurons with 1-to-1 connections
# rlif = snn.RLeaky(beta=beta, all_to_all=False) # initialize RLeaky Neuron
# spk, mem = rlif.init_rleaky() # initialize state variables
# x = torch.rand(1) # generate random input
#
# spk_recording = []
# mem_recording = []
#
# # run simulation
# for step in range(num_steps):
#     spk, mem = rlif(x, spk, mem)
#     spk_recording.append(spk)
#     mem_recording.append(mem)


# disable learning, or use your own initialization variables
# rlif = snn.RLeaky(beta=beta, all_to_all=False, learn_recurrent=False) # disable learning of recurrent connection
# rlif.V = torch.rand(1) # set this to layer size
# print(f"The recurrent weight is: {rlif.V.item()}")


# # Linear feedback
# beta = 0.9 # membrane potential decay rate
# num_steps = 10 # 10 time steps
#
# rlif = snn.RLeaky(beta=beta, linear_features=10)  # initialize RLeaky Neuron
# spk, mem = rlif.init_rleaky() # initialize state variables
# x = torch.rand(10) # generate random input
#
# spk_recording = []
# mem_recording = []
#
# # run simulation
# for step in range(num_steps):
#     spk, mem = rlif(x, spk, mem)
#     spk_recording.append(spk)
#     mem_recording.append(mem)
#
#
# # Convolutional feedback
# beta = 0.9 # membrane potential decay rate
# num_steps = 10 # 10 time steps
#
# rlif = snn.RLeaky(beta=beta, conv2d_channels=3, kernel_size=(5,5))  # initialize RLeaky Neuron
# spk, mem = rlif.init_rleaky() # initialize state variables
# x = torch.rand(3, 32, 32) # generate random 3D input
#
# spk_recording = []
# mem_recording = []
#
# # run simulation
# for step in range(num_steps):
#     spk, mem = rlif(x, spk, mem)
#     spk_recording.append(spk)
#     mem_recording.append(mem)


# Construct Model
class Net(torch.nn.Module):
    """Simple spiking neural network in snntorch."""

    def __init__(self, timesteps, hidden, beta):
        super().__init__()

        self.timesteps = timesteps
        self.hidden = hidden
        self.beta = beta

        # layer 1
        self.fc1 = torch.nn.Linear(in_features=784, out_features=self.hidden).to(device)
        self.rlif1 = snn.RLeaky(beta=self.beta, linear_features=self.hidden).to(device)

        # layer 2
        self.fc2 = torch.nn.Linear(in_features=self.hidden, out_features=10).to(device)
        self.rlif2 = snn.RLeaky(beta=self.beta, linear_features=10).to(device)

    def forward(self, x):
        """Forward pass for several time steps."""

        # Initalize membrane potential
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()
        spk1 = spk1.to(device)
        spk2 = spk2.to(device)
        mem1 = mem1.to(device)
        mem2 = mem2.to(device)

        # Empty lists to record outputs
        spk_recording = []

        for step in range(self.timesteps):
            spk1, mem1 = self.rlif1(self.fc1(x), spk1, mem1)
            spk2, mem2 = self.rlif2(self.fc2(spk1), spk2, mem2)
            spk_recording.append(spk2)

        return torch.stack(spk_recording)


hidden = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = Net(timesteps=num_steps, hidden=hidden, beta=0.9).to(device)

loss_function = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
# 100%|██████████| 5/5 [01:10<00:00, 14.10s/it, loss=1.879e-01]
# The total accuracy on the test set is: 97.06%

#loss_function = SF.mse_membrane_loss(on_target=1.05, off_target=0.2)
# 100%|██████████| 5/5 [01:17<00:00, 15.55s/it, loss=6.008e-01]
# The total accuracy on the test set is: 95.78%

batch_size = 128
data_path='/data/mnist'


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


num_epochs = 5
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
loss_hist = []

with tqdm.trange(num_epochs) as pbar:
    for _ in pbar:
        train_batch = iter(train_loader)
        minibatch_counter = 0
        loss_epoch = []

        for feature, label in train_batch:
            feature = feature.to(device)
            label = label.to(device)

            spk = model(feature.flatten(1)) # forward-pass
            loss_val = loss_function(spk, label) # apply loss
            optimizer.zero_grad() # zero out gradients
            loss_val.backward() # calculate gradients
            optimizer.step() # update weights

            loss_hist.append(loss_val.item())
            minibatch_counter += 1

            avg_batch_loss = sum(loss_hist) / minibatch_counter
            pbar.set_postfix(loss="%.3e" % avg_batch_loss)


test_batch = iter(test_loader)
minibatch_counter = 0
loss_epoch = []

model.eval()
model = model.to(device)
with torch.no_grad():
    total = 0
    acc = 0
    for feature, label in test_batch:
        feature = feature.to(device)
        label = label.to(device)
        # Add flatten.to(device)
        spk = model(feature.flatten(1).to(device)) # forward-pass
        acc += SF.accuracy_rate(spk, label) * spk.size(1)
        total += spk.size(1)

print(f"The total accuracy on the test set is: {(acc/total) * 100:.2f}%")

