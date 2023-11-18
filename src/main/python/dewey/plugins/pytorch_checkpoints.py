import hashlib
import os
import torch

from dewey.core import run_next_plugin


checkpoints_dir = "models"


def run_epoch(plugin_data, next):
    chkp_folder = get_checkpoint_folder(plugin_data)
    model_name = plugin_data.get("model")._get_name()
    chkp_file_name = model_name + str(plugin_data.get("epoch_number")) + ".chkp"
    path = os.path.join(chkp_folder, chkp_file_name)
    checkpoint_exists = os.path.exists(path)
    if checkpoint_exists and plugin_data.get("use_checkpoints"):
        checkpoint = torch.load(path)
        plugin_data.get("model").load_state_dict(checkpoint["model_state_dict"])
        plugin_data.get("optimizer").load_state_dict(checkpoint["optimizer_state_dict"])
        plugin_data.set("training_loss", checkpoint["training_loss"])
        plugin_data.set("validation_loss", checkpoint["validation_loss"])
        plugin_data.set("loaded_checkpoint", True)
        print(f"Epoch {plugin_data.get('epoch_number')+1}: Loaded from checkpoint.")
    run_next_plugin(plugin_data, next)
    if not plugin_data.get("loaded_checkpoint"):
        os.makedirs(chkp_folder, exist_ok=True)
        torch.save({
                    "epoch": plugin_data.get("epoch_number"),
                    "model_state_dict": plugin_data.get("model").state_dict(),
                    "optimizer_state_dict": plugin_data.get("optimizer").state_dict(),
                    "training_loss": plugin_data.get("training_loss"),
                    "validation_loss": plugin_data.get("validation_loss")
                    }, path)
    plugin_data.set("loaded_checkpoint", False)


def get_checkpoint_folder(plugin_data):
    hyperparam_str = str(plugin_data.get("loss")) + str(plugin_data.get("optimizer"))
    hyperparam_hash = hashlib.blake2b(hyperparam_str.encode('utf-8'), digest_size=16).hexdigest()
    return os.path.join(checkpoints_dir, hyperparam_hash)