import hashlib
import os
import json


checkpoints_dir = "models"


def run_epoch(plugin_data, next):
    chkp_folder = get_checkpoint_folder(plugin_data)
    model_name = "model"
    chkp_name = model_name + str(plugin_data.get("epoch_number"))
    path = os.path.join(chkp_folder, chkp_name)
    checkpoint_exists = os.path.exists(path + ".index")
    if checkpoint_exists and plugin_data.get("use_checkpoints"):
        plugin_data.get("model").load_weights(path)
        checkpoint = load_loss(path + ".json")
        plugin_data.set("training_loss", checkpoint["training_loss"])
        plugin_data.set("validation_loss", checkpoint["validation_loss"])
        plugin_data.set("loaded_checkpoint", True)
        print(f"Epoch {plugin_data.get('epoch_number')+1}: Loaded from checkpoint.")
    next(plugin_data)
    if not plugin_data.get("loaded_checkpoint"):
        os.makedirs(chkp_folder, exist_ok=True)
        plugin_data.get("model").save_weights(path)
        save_loss(plugin_data, path + ".json")
    plugin_data.set("loaded_checkpoint", False)


def get_checkpoint_folder(plugin_data):
    hyperparam_str = str(plugin_data.get("loss").get_config()) + str(plugin_data.get("optimizer").get_config())
    hyperparam_hash = hashlib.blake2b(hyperparam_str.encode('utf-8'), digest_size=16).hexdigest()
    return os.path.join(checkpoints_dir, hyperparam_hash)


def save_loss(plugin_data, path):
    loss_dict = {"training_loss": plugin_data.get("training_loss"),
                 "validation_loss": plugin_data.get("validation_loss")}
    with open(path, "w") as f:
        f.write(json.dumps(loss_dict))


def load_loss(path):
    with open(path) as f:
        d = json.load(f)
    return d
