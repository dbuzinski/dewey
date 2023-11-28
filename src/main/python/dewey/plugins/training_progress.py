from alive_progress import alive_bar


def run_training(plugin_data, next):
    print(f"Training - {hyperparams_to_str(plugin_data.get('hyperparameters'))}")
    next(plugin_data)
    print("\n")


def run_epoch(plugin_data, next):
    title = f"Epoch {plugin_data.get('epoch_number')+1}:"
    disable = plugin_data.get("loaded_checkpoint")
    with alive_bar(plugin_data.get("training_data_len"), title=title, dual_line=True, disable=disable) as bar:
        plugin_data.set("progress_bar", bar)
        next(plugin_data)
    training_loss = str(plugin_data.get('training_loss'))
    validation_loss = str(plugin_data.get('validation_loss'))
    print(f"Training Loss: {training_loss}, Validation Loss: {validation_loss}")


def run_training_batch(plugin_data, next):
    bar = plugin_data.get("progress_bar")
    next(plugin_data)
    bar()


def run_backpropegation(plugin_data, next):
    update_loss_ind = plugin_data.get("training_data_len") // 20
    bar = plugin_data.get("progress_bar")
    next(plugin_data)
    if plugin_data.get("batch_number") % update_loss_ind == update_loss_ind - 1:
        bar.text(f"Training Loss: {str(plugin_data.get('training_loss'))}")


def hyperparams_to_str(hyperparams):
    def stringify(val):
        return str(val)

    return ", ".join(list(map(lambda it: f"{it[0]}: {stringify(it[1])}", hyperparams.items())))
