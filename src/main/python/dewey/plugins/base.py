from dewey.core import run_next_plugin


def run_training(plugin_data, next):
    run_next_plugin(plugin_data, next)


def run_epoch(plugin_data, next):
    run_next_plugin(plugin_data, next)


def run_training_batch(plugin_data, next):
    run_next_plugin(plugin_data, next)


def run_backpropegation(plugin_data, next):
    run_next_plugin(plugin_data, next)


def run_validation(plugin_data, next):
    run_next_plugin(plugin_data, next)


def run_validation_batch(plugin_data, next):
    run_next_plugin(plugin_data, next)
