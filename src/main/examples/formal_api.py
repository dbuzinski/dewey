from dewey.ModelTrainer import ModelTrainer
from dewey.DataSpecification import DataSpecification
from dewey.plugins.pytorch.PytorchCheckpointPlugin import PytorchCheckpointPlugin
from dewey.plugins.pytorch.PytorchCorePlugin import PytorchCorePlugin
from dewey.plugins.core.TrainingProgressPlugin import TrainingProgressPlugin
from dewey.plugins.core.TensorBoardPlugin import TensorBoardPlugin
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# Define model
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

# Prep data
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)
data_spec = DataSpecification(training_data=training_loader, validation_data=validation_loader)

# Prep model
model = GarmentClassifier()
loss=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

trainer = ModelTrainer(data_spec)
trainer.load_spec(model, loss, optimizer)
trainer.add_plugin(PytorchCorePlugin())
trainer.add_plugin(PytorchCheckpointPlugin())
trainer.add_plugin(TrainingProgressPlugin())
trainer.add_plugin(TensorBoardPlugin())
trainer.train(total_epochs=15)