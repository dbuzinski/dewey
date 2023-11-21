import subprocess
import sys
from importlib import import_module
from inspect import getmembers, isfunction


class PluginManager:
    __instance = None
    delimiter = ":"

    @staticmethod
    def get_instance():
        if not PluginManager.__instance:
            PluginManager()
        return PluginManager.__instance

    def __init__(self):
        self.plugins = []
        self._required_plugins = []
        self.base_module = import_module("dewey.plugins.base")
        PluginManager.__instance = self

    def require(self, plugin_name):
        self._required_plugins.append(plugin_name)

    def install_plugins(self):
        for plugins in self._required_plugins:
            self.install(plugins)

    def install(self, plugin_name):
        if PluginManager.delimiter in plugin_name:
            ind = plugin_name.index(PluginManager.delimiter)
            package_name = plugin_name[0:ind]
            plugin_module = plugin_name[ind+1:]
            pip_cmd = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', package_name]
            subprocess.run(pip_cmd,
                           stdout=subprocess.DEVNULL)
        else:
            package_name = "pydewey"
            plugin_module = f"dewey.plugins.{plugin_name}"
            extras_name = plugin_name.replace('_', '-')
            pip_cmd = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', f'pydewey[{extras_name}]']
            subprocess.run(pip_cmd,
                           stdout=subprocess.DEVNULL)
        plugin = self.load_plugin_module(plugin_module)
        self.plugins.append(plugin)

    def load_plugin_module(self, plugin_module):
        mod = import_module(plugin_module)
        base_functions = getmembers(self.base_module, isfunction)
        for fn_name, fn in base_functions:
            if not hasattr(mod, fn_name):
                setattr(mod, fn_name, fn)
        return mod

    def run_stage(self, trainer, stage, plugin_data):
        plugin_iterator = reversed([trainer] + self.plugins)
        stage_fns = map(lambda plugin: getattr(plugin, stage), plugin_iterator)

        def run_next_plugin(plugin_data):
            f = next(stage_fns)
            f(plugin_data, run_next_plugin)
        plugin_fn = next(stage_fns)
        plugin_fn(plugin_data, run_next_plugin)
