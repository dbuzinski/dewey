from dewey.core import run_next_plugin


running_loss = 0


def run_epoch(plugin_data, next):
    global running_loss
    running_loss = 0
    run_next_plugin(plugin_data, next)


def run_backpropegation(plugin_data, next):
    global running_loss
    run_next_plugin(plugin_data, next)
    running_loss += plugin_data.get("batch_loss")
    if plugin_data.get("batch_number") % 1000 == 999:
        plugin_data.set("training_loss", running_loss / 1000)
        running_loss = 0


def run_validation(plugin_data, next):
    global running_loss
    running_loss = 0
    run_next_plugin(plugin_data, next)
    plugin_data.set("validation_loss", running_loss / plugin_data.get("validation_data_len"))


def run_validation_batch(plugin_data, next):
    global running_loss
    run_next_plugin(plugin_data, next)
    running_loss += plugin_data.get("batch_loss")
