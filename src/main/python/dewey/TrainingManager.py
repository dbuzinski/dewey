import itertools

from dewey.DataSpecification import DataSpecification
from dewey.core import ModelTrainer


class TrainingManager:
    def __init__(self, model, loss, optimizer, data_spec, total_epochs):
        self.model = model
        self.data_spec = data_spec
        self.optimizer = optimizer
        self.loss = loss
        self.total_epochs = total_epochs

    def train(self):
        trainer = ModelTrainer(self.data_spec)
        for model in self.model:
            for loss, optimizer in itertools.product(self.loss, self.optimizer):
                trainer.load_spec(model, loss, optimizer)
                trainer.train(total_epochs=self.total_epochs)

    @staticmethod
    def from_training_module(module, epochs=1):
        def as_vec(x):
            return x if hasattr(x, "__len__") else [x]

        data_spec = DataSpecification()
        if hasattr(module, "training_data"):
            data_spec.training_data = module.training_data
        else:
            raise Exception("training_data must be specified")
        if hasattr(module, "validation_data"):
            data_spec.validation_data = module.validation_data
        if hasattr(module, "model"):
            model = as_vec(module.model)
        else:
            raise Exception("model must be specified")
        if hasattr(module, "loss"):
            loss = as_vec(module.loss)
        else:
            raise Exception("loss must be specified")
        if hasattr(module, "optimizer"):
            optimizer = as_vec(module.optimizer)
        else:
            raise Exception("optimizer must be specified")
        return TrainingManager(model, loss, optimizer, data_spec, total_epochs=epochs)
