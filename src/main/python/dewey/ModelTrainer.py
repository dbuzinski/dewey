from dewey.internal.PluginManager import PluginManager
from dewey.internal.PluginData import PluginData
from dewey.internal.TrainingOperator import TrainingOperator


class ModelTrainer(TrainingOperator):
    def __init__(self, data_spec, device=None):
        super().__init__()
        self.training_spec = dict()
        self.data_spec = data_spec
        self.device = device
        self.plugin_manager = PluginManager()

    @property
    def plugins(self):
        return self.plugin_manager.plugins

    def load_spec(self, model, loss, optimizer):
        self.training_spec["model"] = model
        self.training_spec["loss"] = loss
        self.training_spec["optimizer"] = optimizer

    def train(self, total_epochs=1):
        # plugin_data = PluginData.from_spec(self.training_spec)
        plugin_data = PluginData()
        plugin_data.set("model", self.training_spec["model"])
        plugin_data.set("loss", self.training_spec["loss"])
        plugin_data.set("optimizer", self.training_spec["optimizer"])
        plugin_data.set("total_epochs", total_epochs)
        self.plugin_manager.run_on_plugins(self, "run_training", plugin_data)

    def run_training(self, plugin_data):
        plugin_data.set("training_data_len", len(self.data_spec.training_data))
        plugin_data.set("validation_data_len", len(self.data_spec.validation_data))
        for epoch in range(plugin_data.get("total_epochs")):
            plugin_data.set("epoch_number", epoch)
            self.plugin_manager.run_on_plugins(self, "run_epoch", plugin_data)

    def run_epoch(self, plugin_data):
        if not plugin_data.get("loaded_checkpoint"):
            for batch_number, (batch_data, batch_labels) in enumerate(self.data_spec.training_data):
                plugin_data.prepare_batch(batch_number, batch_data, batch_labels)
                self.plugin_manager.run_on_plugins(self, "run_training_batch", plugin_data)
            if plugin_data.get("validation_data_len"):
                self.plugin_manager.run_on_plugins(self, "run_validation", plugin_data)

    def run_training_batch(self, plugin_data):
        self.plugin_manager.run_on_plugins(self, "run_backpropegation", plugin_data)

    def run_backpropegation(self, plugin_data):
        pass

    def run_validation(self, plugin_data):
        for batch_number, (batch_data, batch_labels) in enumerate(self.data_spec.validation_data):
            plugin_data.prepare_batch(batch_number, batch_data, batch_labels)
            self.plugin_manager.run_on_plugins(self, "run_validation_batch", plugin_data)

    def run_validation_batch(self, plugin_data):
        pass

    def add_plugin(self, plugin):
        self.plugin_manager.add_plugin(plugin)
