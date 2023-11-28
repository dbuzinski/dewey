import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dewey.core import use_plugin

use_plugin("pytorch")
use_plugin("training_progress")
use_plugin("pytorch_checkpoints")


# set seed
random.seed(0)
torch.manual_seed(0)


# hyperparameters
hyperparameters = {
    "batch_size": [32, 64],
    "learning_rate": [0.001, 0.0005],
    "momentum": 0.9,
}


# prep data
def data(hyperparams):
    batch_size = hyperparams["batch_size"]
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    training_set = torchvision.datasets.FashionMNIST('./data',
                                                     train=True,
                                                     transform=transform,
                                                     download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data',
                                                       train=False,
                                                       transform=transform,
                                                       download=True)
    training_data = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_data = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    return {"training_data": training_data, "validation_data": validation_data}


class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define model
model = GarmentClassifier()


# Define loss
loss = torch.nn.CrossEntropyLoss()


# Define optimizer
def optimizer(hyperparams):
    lr = hyperparams["learning_rate"]
    momentum = hyperparams["momentum"]
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
