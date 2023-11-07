from dewey.internal.TrainingOperator import TrainingOperator


class ModelTrainerPlugin(TrainingOperator):
    def __init__(self):
        super().__init__()

    def run_training(self, plugin_data):
        next_plugin = next(self.operator_iterator, None)
        next_plugin and next_plugin.run_training(plugin_data)

    def run_epoch(self, plugin_data):
        next_plugin = next(self.operator_iterator, None)
        next_plugin and next_plugin.run_epoch(plugin_data)

    def run_training_batch(self, plugin_data):
        next_plugin = next(self.operator_iterator, None)
        next_plugin and next_plugin.run_training_batch(plugin_data)

    def run_backpropegation(self, plugin_data):
        next_plugin = next(self.operator_iterator, None)
        next_plugin and next_plugin.run_backpropegation(plugin_data)

    def run_validation(self, plugin_data):
        next_plugin = next(self.operator_iterator, None)
        next_plugin and next_plugin.run_validation(plugin_data)

    def run_validation_batch(self, plugin_data):
        next_plugin = next(self.operator_iterator, None)
        next_plugin and next_plugin.run_validation_batch(plugin_data)
