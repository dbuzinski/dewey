from dewey.ModelTrainerPlugin import ModelTrainerPlugin
from alive_progress import alive_bar


class TrainingProgressPlugin(ModelTrainerPlugin):
    def __init__(self):
        super().__init__()
        self.bar = None

    def run_epoch(self, plugin_data):
        title = f"Epoch {plugin_data.get('epoch_number')+1}:"
        disable = plugin_data.get("loaded_checkpoint")
        with alive_bar(plugin_data.get("training_data_len"), title=title, dual_line=True, disable=disable) as bar:
            self.bar = bar
            super().run_epoch(plugin_data)
        training_loss = str(plugin_data.get('training_loss'))
        validation_loss = str(plugin_data.get('validation_loss'))
        print(f"Training Loss: {training_loss}, Validation Loss: {validation_loss}")

    def run_training_batch(self, plugin_data):
        super().run_training_batch(plugin_data)
        self.bar()

    def run_backpropegation(self, plugin_data):
        super().run_backpropegation(plugin_data)
        if plugin_data.get("batch_number") % 1000 == 999:
            self.bar.text(f"Training Loss: {str(plugin_data.get('training_loss'))}")
