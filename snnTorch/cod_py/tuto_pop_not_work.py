import torch, torch.nn as nn
import snntorch as snn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from snntorch import surrogate
import snntorch.functional as SF
from snntorch import backprop


batch_size = 128
data_path='/tmp/data/fmnist'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

fmnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
fmnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(fmnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(fmnist_test, batch_size=batch_size, shuffle=True)

# network parameters
num_inputs = 28*28
num_hidden = 128
num_outputs = 10
num_steps = 1

# spiking neuron parameters
beta = 0.9  # neuron decay rate
grad = surrogate.fast_sigmoid()

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(num_inputs, num_hidden),
                    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                    nn.Linear(num_hidden, num_outputs),
                    snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
                    ).to(device)

pop_outputs = 500

net_pop = nn.Sequential(nn.Flatten(),
                        nn.Linear(num_inputs, num_hidden),
                        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
                        nn.Linear(num_hidden, pop_outputs),
                        snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
                        ).to(device)


optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0)

from snntorch import utils

def test_accuracy(data_loader, net, num_steps, population_code=False, num_classes=False):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()

    data_loader = iter(data_loader)
    for data, targets in data_loader:
      data = data.to(device)
      targets = targets.to(device)
      utils.reset(net)
      spk_rec, _ = net(data)

      if population_code:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets, population_code=True, num_classes=10) * spk_rec.size(1)
      else:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets) * spk_rec.size(1)

      total += spk_rec.size(1)

  return acc/total


num_epochs = 5

# training loop
for epoch in range(num_epochs):

    avg_loss = backprop.BPTT(net, train_loader, num_steps=num_steps,
                          optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)

    print(f"Epoch: {epoch}")
    print(f"Test set accuracy: {test_accuracy(test_loader, net, num_steps)*100:.3f}%\n")

loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=10)
optimizer = torch.optim.Adam(net_pop.parameters(), lr=2e-3, betas=(0.9, 0.999))

