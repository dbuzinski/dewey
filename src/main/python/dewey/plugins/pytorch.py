import torch

from dewey.core import run_next_plugin


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_training(plugin_data, next):
    plugin_data.get("model").to(device)
    run_next_plugin(plugin_data, next)


def run_epoch(plugin_data, next):
    plugin_data.get("model").train()
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


def run_validation(plugin_data, next):
    plugin_data.get("model").eval()
    with torch.no_grad():
        run_next_plugin(plugin_data, next)


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
