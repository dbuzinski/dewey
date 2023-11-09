class PluginData:
    def __init__(self, use_checkpoints=True):
        self.data = dict()
        self.data["use_checkpoints"] = use_checkpoints
        self.data["loaded_checkpoint"] = False

    def set(self, k, v):
        self.data[k] = v

    def get(self, k):
        return self.data.get(k, None)

    def prepare_batch(self, batch_number, batch_data, batch_labels):
        self.data["batch_number"] = batch_number
        self.data["batch_data"] = batch_data
        self.data["batch_labels"] = batch_labels
