import argparse
import importlib.util
import os
import sys

from dewey.internal.PluginManager import PluginManager
from dewey.TrainingManager import TrainingManager


def main():
    parser = argparse.ArgumentParser(
        description="A tool to help train machine learning models"
        )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="N",
        type=int,
        help="number of training epochs")
    args = parser.parse_args()

    # Import training spec
    train_path = "train.py"
    train_dir = os.path.dirname(os.path.abspath(train_path))
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)

    spec = importlib.util.spec_from_file_location("train", train_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    plugin_manager = PluginManager.get_instance()
    plugin_manager.install_plugins()

    epochs = args.epochs
    trainer = TrainingManager.from_training_module(module, epochs=epochs)
    trainer.train()
