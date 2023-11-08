import itertools
from dewey.DataSpecification import DataSpecification
from dewey.ModelTrainer import ModelTrainer
from dewey.plugins.core.TensorBoardPlugin import TensorBoardPlugin
from dewey.plugins.core.TrainingProgressPlugin import TrainingProgressPlugin
from dewey.plugins.pytorch.PytorchCorePlugin import PytorchCorePlugin
from dewey.plugins.pytorch.PytorchCheckpointPlugin import PytorchCheckpointPlugin

class TrainingManager:
    def __init__(self, model, loss, optimizer, data_spec, total_epochs):
        self.model = model
        self.data_spec = data_spec
        self.optimizer = optimizer
        self.loss = loss
        self.total_epochs = total_epochs

    def train(self):
        trainer = ModelTrainer(self.data_spec)
        trainer.add_plugin(PytorchCorePlugin())
        trainer.add_plugin(PytorchCheckpointPlugin())
        trainer.add_plugin(TrainingProgressPlugin())
        trainer.add_plugin(TensorBoardPlugin())
        for model, loss, optimizer in itertools.product(self.model, self.loss, self.optimizer):
            trainer.load_spec(model, loss, optimizer)
            trainer.train(total_epochs=self.total_epochs)

    @staticmethod
    def from_training_module(module):
        data_spec = DataSpecification()
        if hasattr(module, "training_data"):
            data_spec.training_data = module.training_data
        else:
            raise Exception("training_data must be specified")
        if hasattr(module, "validation_data"):
            data_spec.validation_data = module.validation_data
        if hasattr(module, "model"):
            model = module.model
        else:
            raise Exception("model must be specified")
        if hasattr(module, "loss"):
            loss = module.loss
        else:
            raise Exception("loss must be specified")
        if hasattr(module, "optimizer"):
            optimizer = module.optimizer
        else:
            raise Exception("optimizer must be specified")
        if hasattr(module, "epochs"):
            total_epochs = module.epochs
        else:
            total_epochs = 1
        return TrainingManager(model, loss, optimizer, data_spec, total_epochs=total_epochs)