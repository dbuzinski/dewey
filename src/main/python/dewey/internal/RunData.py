class RunData:
    def __init__(self):
        self.total_epochs = 0
        self.epoch_number = 0
        self.training_data_len = 0
        self.validation_data_len = 0
        self.batch_number = 0
        self.batch_data = None
        self.batch_labels = None
        self.batch_predictions = None
        self.batch_loss = 0

    def prepare_batch(self, batch_number, batch_data, batch_labels):
        self.batch_number = batch_number
        self.batch_data = batch_data
        self.batch_labels = batch_labels

    def _clear_batch_data(self):
        self.batch_data = None
        self.batch_labels = None
        self.batch_predictions = None
        self.batch_loss = 0
