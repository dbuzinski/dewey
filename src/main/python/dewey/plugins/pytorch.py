import copy
import random
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dependencies = ["torch>=2.1.0"]


def run_training(plugin_data, next):
    model = plugin_data.get("model")
    initial_state_dict = copy.deepcopy(model.state_dict())
    model.to(device)
    next(plugin_data)
    model.load_state_dict(initial_state_dict)


def run_epoch(plugin_data, next):
    plugin_data.get("model").train()
    plugin_data.set("running_loss", 0)
    next(plugin_data)
    random.seed(0)
    torch.manual_seed(0)


def run_training_batch(plugin_data, next):
    batch_data = plugin_data.get("batch_data").to(device)
    model = plugin_data.get("model")
    plugin_data.set("batch_predictions", model(batch_data))
    next(plugin_data)


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
    next(plugin_data)
    update_backpropegation_loss(plugin_data)


def run_validation(plugin_data, next):
    plugin_data.get("model").eval()
    plugin_data.set("running_loss", 0)
    with torch.no_grad():
        next(plugin_data)
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
    next(plugin_data)
    update_validation_loss(plugin_data)


def update_backpropegation_loss(plugin_data):
    update_loss_ind = plugin_data.get("training_data_len") // 20
    running_loss = plugin_data.get("running_loss")
    plugin_data.set("running_loss", running_loss + plugin_data.get("batch_loss"))
    if plugin_data.get("batch_number") % update_loss_ind == update_loss_ind - 1:
        plugin_data.set("training_loss", plugin_data.get("running_loss") / update_loss_ind)
        plugin_data.set("running_loss", 0)


def update_validation_loss(plugin_data):
    running_loss = plugin_data.get("running_loss")
    plugin_data.set("running_loss", running_loss + plugin_data.get("batch_loss"))
