def run_training(plugin_data, next):
    next(plugin_data)


def run_epoch(plugin_data, next):
    next(plugin_data)


def run_training_batch(plugin_data, next):
    next(plugin_data)


def run_backpropegation(plugin_data, next):
    next(plugin_data)


def run_validation(plugin_data, next):
    next(plugin_data)


def run_validation_batch(plugin_data, next):
    next(plugin_data)
