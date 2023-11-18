from inspect import currentframe
from dewey.internal.PluginManager import PluginManager


def use_plugin(plugin_name):
    plugin_manager = PluginManager.get_instance()
    plugin_manager.require(plugin_name)


def run_next_plugin(plugin_data, plugin_iterator):
    next_plugin = next(plugin_iterator, None)
    if next_plugin:
        frame = currentframe().f_back
        calling_function_name = frame.f_code.co_name
        next_function = getattr(next_plugin, calling_function_name)
        next_function(plugin_data, plugin_iterator)
