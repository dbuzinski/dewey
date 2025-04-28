import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dewey.core import use_plugin
from model import GarmentClassifier

use_plugin("pytorch")
use_plugin("training_progress")
use_plugin("pytorch_checkpoints")

# Set seed
random.seed(0)
torch.manual_seed(0)

# Define hyperparameters
hyperparameters = {
    "batch_size": [512],
    "learning_rate": [0.001, 0.0005],
}

# Define dataset
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


# Define model
model = GarmentClassifier()

# Define loss
loss = torch.nn.CrossEntropyLoss()

# Define optimizer
def optimizer(hyperparams):
    lr = hyperparams["learning_rate"]
    return torch.optim.Adam(model.parameters(), lr=lr)
