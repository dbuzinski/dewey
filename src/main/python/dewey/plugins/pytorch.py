import torch

from dewey.core import run_next_plugin


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_training(plugin_data, next):
    plugin_data.get("model").to(device)
    run_next_plugin(plugin_data, next)


def run_epoch(plugin_data, next):
    plugin_data.get("model").train()
    plugin_data.set("running_loss", 0)
    run_next_plugin(plugin_data, next)


def run_training_batch(plugin_data, next):
    batch_data = plugin_data.get("batch_data").to(device)
    model = plugin_data.get("model")
    plugin_data.set("batch_predictions", model(batch_data))
    run_next_plugin(plugin_data, next)


def run_backpropegation(plugin_data, next):
    batch_predictions = plugin_data.get("batch_predictions")
    batch_labels = plugin_data.get("batch_labels").to(device)
    loss_fn = plugin_data.get("loss")
    optimizer = plugin_data.get("optimizer")
    optimizer.zero_grad()
    loss = loss_fn(batch_predictions, batch_labels)
    loss.backward()
    optimizer.step()
    plugin_data.set("batch_loss", loss.detach().item())
    run_next_plugin(plugin_data, next)
    update_backpropegation_loss(plugin_data)


def run_validation(plugin_data, next):
    plugin_data.get("model").eval()
    plugin_data.set("running_loss", 0)
    with torch.no_grad():
        run_next_plugin(plugin_data, next)
    plugin_data.set("validation_loss", plugin_data.get("running_loss") / plugin_data.get("validation_data_len"))


def run_validation_batch(plugin_data, next):
    model = plugin_data.get("model")
    batch_data = plugin_data.get("batch_data")
    batch_labels = plugin_data.get("batch_labels")
    batch_predictions = model(batch_data)
    plugin_data.set("batch_predictions", batch_predictions)
    loss_fn = plugin_data.get("loss")
    loss = loss_fn(batch_predictions, batch_labels)
    plugin_data.set("batch_loss", loss.detach().item())
    run_next_plugin(plugin_data, next)
    update_validation_loss(plugin_data)


def update_backpropegation_loss(plugin_data):
    running_loss = plugin_data.get("running_loss")
    plugin_data.set("running_loss", running_loss + plugin_data.get("batch_loss"))
    if plugin_data.get("batch_number") % 1000 == 999:
        plugin_data.set("training_loss", plugin_data.get("running_loss") / 1000)
        plugin_data.set("running_loss", 0)


def update_validation_loss(plugin_data):
    running_loss = plugin_data.get("running_loss")
    plugin_data.set("running_loss", running_loss + plugin_data.get("batch_loss"))
