from dewey.ModelTrainerPlugin import ModelTrainerPlugin
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorBoardPlugin(ModelTrainerPlugin):
    def __init__(self, model_name="model"):
        super().__init__()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(f"runs/{model_name}_{timestamp}")

    def run_epoch(self, plugin_data):
        super().run_epoch(plugin_data)
        self.writer.add_scalars("Training vs. Validation Loss",
                                {"Training": plugin_data.get("training_loss"),
                                 "Validation": plugin_data.get("validation_loss")},
                                plugin_data.get("epoch_number") + 1)
        self.writer.flush()

    def run_backpropegation(self, plugin_data):
        super().run_backpropegation(plugin_data)
        if plugin_data.get("batch_number") % 1000 == 999:
            t = plugin_data.get("epoch_number")*plugin_data.get("training_data_len")+plugin_data.get("batch_number")+1
            self.writer.add_scalar("Loss/train", plugin_data.get("training_loss"), t)
