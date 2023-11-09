from dewey.ModelTrainerPlugin import ModelTrainerPlugin
import torch


class PytorchCorePlugin(ModelTrainerPlugin):
    def __init__(self, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.device = device
        print(f"Training on device {device.type}.")

    def run_training(self, plugin_data):
        self.model = plugin_data.get("model").to(self.device)
        self.loss_fn = plugin_data.get("loss")
        self.optimizer = plugin_data.get("optimizer")
        super().run_training(plugin_data)

    def run_epoch(self, plugin_data):
        self.model.train()
        super().run_epoch(plugin_data)

    def run_training_batch(self, plugin_data):
        batch_data = plugin_data.get("batch_data").to(self.device)
        plugin_data.set("batch_predictions", self.model(batch_data))
        super().run_training_batch(plugin_data)

    def run_backpropegation(self, plugin_data):
        batch_predictions = plugin_data.get("batch_predictions")
        batch_labels = plugin_data.get("batch_labels").to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss_fn(batch_predictions, batch_labels)
        loss.backward()
        self.optimizer.step()
        plugin_data.set("batch_loss", loss.detach().item())
        super().run_backpropegation(plugin_data)

    def run_validation(self, plugin_data):
        self.model.eval()
        with torch.no_grad():
            super().run_validation(plugin_data)

    def run_validation_batch(self, plugin_data):
        batch_data = plugin_data.get("batch_data")
        batch_labels = plugin_data.get("batch_labels")
        batch_predictions = self.model(batch_data)
        plugin_data.set("batch_predictions", batch_predictions)
        loss = self.loss_fn(batch_predictions, batch_labels)
        plugin_data.set("batch_loss", loss.detach().item())
        super().run_validation_batch(plugin_data)
