from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


model_name = "model"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f"runs/{model_name}_{timestamp}")


def run_epoch(plugin_data, next):
    next(plugin_data)
    writer.add_scalars("Training vs. Validation Loss",
                       {"Training": plugin_data.get("training_loss"),
                        "Validation": plugin_data.get("validation_loss")},
                       plugin_data.get("epoch_number") + 1)
    writer.flush()


def run_backpropegation(plugin_data, next):
    next(plugin_data)
    if plugin_data.get("batch_number") % 1000 == 999:
        t = plugin_data.get("epoch_number")*plugin_data.get("training_data_len")+plugin_data.get("batch_number")+1
        writer.add_scalar("Loss/train", plugin_data.get("training_loss"), t)
