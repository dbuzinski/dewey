import itertools
import types

from dewey.core import ModelTrainer


class TrainingManager:
    def __init__(self, hyperparameters, data, model, loss, optimizer):
        self.hyperparameters = hyperparameters
        self.data = data
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def train(self):
        trainer = ModelTrainer()
        hyperparameters = self.hyperparameters
        if not hyperparameters:
            hyperparameters = {"epochs": 1}
        hyperparams = _vals_to_lists(hyperparameters)
        keys, values = zip(*hyperparams.items())
        hyperparam_combs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for h in hyperparam_combs:
            data = _load_from_hyperparameters(self.data, h)
            model = _load_from_hyperparameters(self.model, h)
            loss = _load_from_hyperparameters(self.loss, h)
            optimizer = _load_from_hyperparameters(self.optimizer, h)
            trainer.load_spec(hyperparams, data, model, loss, optimizer)
            trainer.train()

    @staticmethod
    def from_training_module(module, epochs=None):
        hyperparameters = _load_optional_from_module(module, "hyperparameters", {})
        if epochs:
            hyperparameters["epochs"] = epochs
        data = _load_required_from_module(module, "data")
        model = _load_required_from_module(module, "model")
        loss = _load_required_from_module(module, "loss")
        optimizer = _load_required_from_module(module, "optimizer")
        return TrainingManager(hyperparameters, data, model, loss, optimizer)


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
