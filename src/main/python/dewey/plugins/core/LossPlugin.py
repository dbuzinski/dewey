from dewey.ModelTrainerPlugin import ModelTrainerPlugin


class LossPlugin(ModelTrainerPlugin):
    def __init__(self):
        super().__init__()
        self.running_loss = 0

    def run_epoch(self, plugin_data):
        self.running_loss = 0
        super().run_epoch(plugin_data)

    def run_backpropegation(self, plugin_data):
        super().run_backpropegation(plugin_data)
        self.running_loss += plugin_data.get("batch_loss")
        if plugin_data.get("batch_number") % 1000 == 999:
            plugin_data.set("training_loss", self.running_loss / 1000)
            self.running_loss = 0

    def run_validation(self, plugin_data):
        self.running_loss = 0
        super().run_validation(plugin_data)
        plugin_data.set("validation_loss", self.running_loss / plugin_data.get("validation_data_len"))

    def run_validation_batch(self, plugin_data):
        super().run_validation_batch(plugin_data)
        self.running_loss += plugin_data.get("batch_loss")
