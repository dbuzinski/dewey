from dewey.ModelTrainerPlugin import ModelTrainerPlugin
from alive_progress import alive_bar


class TrainingProgressPlugin(ModelTrainerPlugin):
    def __init__(self):
        super().__init__()
        self.bar = None
        self.running_loss = 0
        self.training_loss = 0
        self.validation_loss = 0

    def run_epoch(self, plugin_data):
        self.running_loss = 0
        title = f"Epoch {plugin_data.get('epoch_number')+1}:"
        with alive_bar(plugin_data.get("training_data_len"), title=title, dual_line=True) as bar:
            self.bar = bar
            super().run_epoch(plugin_data)
        print(f"Training Loss: {str(self.training_loss)}, Validation Loss: {str(self.validation_loss)}")

    def run_training_batch(self, plugin_data):
        super().run_training_batch(plugin_data)
        self.bar()

    def run_backpropegation(self, plugin_data):
        super().run_backpropegation(plugin_data)
        self.running_loss += plugin_data.get("batch_loss")
        if plugin_data.get("batch_number") % 1000 == 999:
            self.training_loss = self.running_loss / 1000
            self.running_loss = 0
            self.bar.text(f"Training Loss: {str(self.training_loss)}")

    def run_validation(self, plugin_data):
        self.running_loss = 0
        super().run_validation(plugin_data)
        self.validation_loss = self.running_loss / plugin_data.get("validation_data_len")

    def run_validation_batch(self, plugin_data):
        super().run_validation_batch(plugin_data)
        self.running_loss += plugin_data.get("batch_loss")
