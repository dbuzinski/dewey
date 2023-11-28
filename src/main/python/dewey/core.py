from dewey.internal.PluginManager import PluginManager
from dewey.internal.PluginData import PluginData


def use_plugin(plugin_name):
    plugin_manager = PluginManager.get_instance()
    plugin_manager.require(plugin_name)


class ModelTrainer:
    def __init__(self):
        super().__init__()
        self.training_spec = dict()
        self.plugin_manager = PluginManager.get_instance()

    @property
    def plugins(self):
        return self.plugin_manager.plugins

    def load_spec(self, hyperparameters, data, model, loss, optimizer):
        self.training_spec["hyperparameters"] = hyperparameters
        self.training_spec["data"] = data
        self.training_spec["model"] = model
        self.training_spec["loss"] = loss
        self.training_spec["optimizer"] = optimizer

    def train(self, total_epochs=1):
        plugin_data = PluginData()
        plugin_data.set("model", self.training_spec["model"])
        plugin_data.set("loss", self.training_spec["loss"])
        plugin_data.set("optimizer", self.training_spec["optimizer"])
        plugin_data.set("hyperparameters", self.training_spec["hyperparameters"])
        plugin_data.set("total_epochs", total_epochs)
        self.plugin_manager.run_stage(self, "run_training", plugin_data)

    def run_training(self, plugin_data, next):
        plugin_data.set("training_data_len", len(self.training_spec.get("data").get("training_data")))
        plugin_data.set("validation_data_len", len(self.training_spec.get("data").get("validation_data", [])))
        for epoch in range(plugin_data.get("total_epochs")):
            plugin_data.set("epoch_number", epoch)
            self.plugin_manager.run_stage(self, "run_epoch", plugin_data)

    def run_epoch(self, plugin_data, next):
        if not plugin_data.get("loaded_checkpoint"):
            data = self.training_spec.get("data").get("training_data")
            for batch_number, (batch_data, batch_labels) in enumerate(data):
                plugin_data.prepare_batch(batch_number, batch_data, batch_labels)
                self.plugin_manager.run_stage(self, "run_training_batch", plugin_data)
            if plugin_data.get("validation_data_len"):
                self.plugin_manager.run_stage(self, "run_validation", plugin_data)

    def run_training_batch(self, plugin_data, next):
        self.plugin_manager.run_stage(self, "run_backpropegation", plugin_data)

    def run_backpropegation(self, plugin_data, next):
        pass

    def run_validation(self, plugin_data, next):
        data = self.training_spec.get("data").get("validation_data", [])
        for batch_number, (batch_data, batch_labels) in enumerate(data):
            plugin_data.prepare_batch(batch_number, batch_data, batch_labels)
            self.plugin_manager.run_stage(self, "run_validation_batch", plugin_data)

    def run_validation_batch(self, plugin_data, next):
        pass
