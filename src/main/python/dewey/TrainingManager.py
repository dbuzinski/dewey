import itertools
import types

from dewey.core import ModelTrainer


class TrainingManager:
    def __init__(self, hyperparameters, data, model, loss, optimizer, total_epochs=1):
        self.hyperparameters = hyperparameters
        self.data = data
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.total_epochs = total_epochs

    def train(self):
        trainer = ModelTrainer()
        hyperparameters = self.hyperparameters
        hyperparams = _vals_to_lists(hyperparameters)
        keys, values = zip(*hyperparams.items())
        hyperparam_combs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for h in hyperparam_combs:
            data = _load_from_hyperparameters(self.data, h)
            model = _load_from_hyperparameters(self.model, h)
            loss = _load_from_hyperparameters(self.loss, h)
            optimizer = _load_from_hyperparameters(self.optimizer, h)
            trainer.load_spec(h, data, model, loss, optimizer)
            trainer.train(total_epochs=self.total_epochs)

    @staticmethod
    def from_training_module(module, epochs=None):
        epochs = epochs or 1
        hyperparameters = _load_optional_from_module(module, "hyperparameters", {})
        data = _load_required_from_module(module, "data")
        model = _load_required_from_module(module, "model")
        loss = _load_required_from_module(module, "loss")
        optimizer = _load_required_from_module(module, "optimizer")
        return TrainingManager(hyperparameters, data, model, loss, optimizer, total_epochs=epochs)


def _vals_to_lists(dict):
    def as_vec(x):
        return x if hasattr(x, "__len__") else [x]

    d = {}
    for k in dict.keys():
        d[k] = as_vec(dict[k])
    return d


def _load_optional_from_module(module, attr, default):
    return getattr(module, attr, default)


def _load_required_from_module(module, attr):
    val = None
    try:
        val = getattr(module, attr)
    except AttributeError:
        raise Exception(f"{attr} must be specified")
    return val


def _load_from_hyperparameters(attr, hyperparams):
    val = attr
    if isinstance(attr, types.FunctionType):
        val = attr(hyperparams)
    return val
