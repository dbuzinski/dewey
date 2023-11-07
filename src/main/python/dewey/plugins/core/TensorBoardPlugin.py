from dewey.ModelTrainerPlugin import ModelTrainerPlugin
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorBoardPlugin(ModelTrainerPlugin):
    def __init__(self, model_name="model"):
        super().__init__()
        self.running_loss = 0
        self.training_loss = 0
        self.validation_loss = 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f"runs/{model_name}_{timestamp}")

    def run_epoch(self, plugin_data):
        self.running_loss = 0
        super().run_epoch(plugin_data)
        self.writer.add_scalars('Training vs. Validation Loss',
                                {'Training': self.training_loss, 'Validation': self.validation_loss},
                                plugin_data.get("epoch_number") + 1)
        self.writer.flush()

    def run_backpropegation(self, plugin_data):
        super().run_backpropegation(plugin_data)
        self.running_loss += plugin_data.get("batch_loss")
        if plugin_data.get("batch_number") % 1000 == 999:
            self.training_loss = self.running_loss / 1000
            self.running_loss = 0
            t = plugin_data.get("epoch_number") * plugin_data.get("training_data_len") + plugin_data.get("batch_number") + 1
            self.writer.add_scalar('Loss/train', self.training_loss, t)

    def run_validation_batch(self, plugin_data):
        super().run_validation_batch(plugin_data)
        self.running_loss += plugin_data.get("batch_loss")

    def run_validation(self, plugin_data):
        self.running_loss = 0
        super().run_validation(plugin_data)
        self.validation_loss = self.running_loss / plugin_data.get("validation_data_len")
