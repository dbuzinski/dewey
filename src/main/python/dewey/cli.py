import argparse
import importlib
import importlib.util

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
    spec = importlib.util.spec_from_file_location("train", "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    epochs = args.epochs or module.epochs
    trainer = TrainingManager.from_training_module(module, epochs=epochs)
    trainer.train()
