class PluginManager:
    def __init__(self):
        self.plugins = []

    def add_plugin(self, plugin):
        self.plugins.append(plugin)

    def run_on_plugins(self, base_operator, method_name, plugin_data):
        operators = [base_operator] + self.plugins
        operator_iter = reversed(operators)
        for i in range(len(operators)):
            operators[i].set_iterator(operator_iter)
        plugin = next(operator_iter)
        plugin_method = getattr(plugin, method_name)
        plugin_method(plugin_data)
