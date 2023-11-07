import itertools
from dewey.ModelTrainer import ModelTrainer

class TrainingManager:
    def __init__(self, model, data, loss, optimizer, total_epochs):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.total_epochs = total_epochs

    def train(self):
        trainer = ModelTrainer()
        for model, loss, optimizer in itertools.product(self.model, self.loss, self.optimizer):
            trainer.load_spec(model, data, loss, optimizer)
            trainer.train(total_epochs=self.total_epochs)

    @staticmethod
    def from_training_module(module):
        if hasattr(module, "model"):
            model = module.model
        else:
            raise Exception("model must be specified")
        if hasattr(module, "data"):
            data = module.data
        else:
            raise Exception("data must be specified")
        if hasattr(module, "loss"):
            loss = module.loss
        else:
            raise Exception("loss must be specified")
        if hasattr(module, "optimizer"):
            optimizer = module.optimizer
        else:
            raise Exception("optimizer must be specified")
        if hasattr(total_epochs, "epochs"):
            total_epochs = module.epochs
        else:
            total_epochs = 1
        return TrainingManager(model, loss, data, optimizer, total_epochs=total_epochs)