from abc import ABC, abstractmethod


class TrainingOperator(ABC):
    def __init__(self):
        self.operator_iterator = enumerate([])

    def set_iterator(self, iter):
        self.operator_iterator = iter

    @abstractmethod
    def run_training(self, plugin_data):
        ...

    @abstractmethod
    def run_epoch(self, plugin_data):
        ...

    @abstractmethod
    def run_training_batch(self, plugin_data):
        ...

    @abstractmethod
    def run_backpropegation(self, plugin_data):
        ...

    @abstractmethod
    def run_validation(self, plugin_data):
        ...

    @abstractmethod
    def run_validation_batch(self, plugin_data):
        ...
