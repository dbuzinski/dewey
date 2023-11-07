from dewey.ModelTrainerPlugin import ModelTrainerPlugin
import os
import torch

class PytorchCheckpointPlugin(ModelTrainerPlugin):
    def __init__(self, checkpoints_dir="models", model_name="model"):
        super().__init__()
        self.checkpoints_dir = checkpoints_dir
        self.model_name = model_name

    def run_epoch(self, plugin_data):
        folder = os.path.join(self.checkpoints_dir, self.model_name, str(plugin_data.get("epoch_number")))
        path = os.path.join(folder, "chkp.pt")
        checkpoint_exists = os.path.exists(path)
        if checkpoint_exists and plugin_data.get("use_checkpoints"):
            checkpoint = torch.load(path)
            plugin_data.get("model").load_state_dict(checkpoint['model_state_dict'])
            plugin_data.get("optimizer").load_state_dict(checkpoint['optimizer_state_dict'])
            return
        super().run_epoch(plugin_data)
        os.makedirs(folder, exist_ok=True)
        torch.save({
                    'epoch': plugin_data.get("epoch_number"),
                    'model_state_dict': plugin_data.get("model").state_dict(),
                    'optimizer_state_dict': plugin_data.get("optimizer").state_dict(),
                    }, path)
