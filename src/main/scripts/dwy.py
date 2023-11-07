#!/usr/bin/env python
if __name__ == '__main__':
    import importlib
    import importlib.util
    from dewey.TrainingManager import TrainingManager

    spec = importlib.util.spec_from_file_location("train", "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer = TrainingManager.from_training_module(module)
    trainer.train()
