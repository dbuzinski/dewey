from alive_progress import alive_bar

from dewey.core import run_next_plugin


bar = None


def run_epoch(plugin_data, next):
    global bar
    title = f"Epoch {plugin_data.get('epoch_number')+1}:"
    disable = plugin_data.get("loaded_checkpoint")
    with alive_bar(plugin_data.get("training_data_len"), title=title, dual_line=True, disable=disable) as prog_bar:
        bar = prog_bar
        run_next_plugin(plugin_data, next)
    training_loss = str(plugin_data.get('training_loss'))
    validation_loss = str(plugin_data.get('validation_loss'))
    print(f"Training Loss: {training_loss}, Validation Loss: {validation_loss}")


def run_training_batch(plugin_data, next):
    global bar
    run_next_plugin(plugin_data, next)
    bar()


def run_backpropegation(plugin_data, next):
    global bar
    run_next_plugin(plugin_data, next)
    if plugin_data.get("batch_number") % 1000 == 999:
        bar.text(f"Training Loss: {str(plugin_data.get('training_loss'))}")
