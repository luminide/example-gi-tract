import numpy as np
import pandas as pd

class LossHistory:
    def __init__(self):
        self.data = []

    def add_val_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, np.nan, val, np.nan])

    def add_train_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, val, np.nan, np.nan])

    def add_epoch_val_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, np.nan, np.nan, val])

    def save(self):
        columns = [
            'index', 'epoch', 'sample_count',
            'train_loss', 'val_loss', 'epoch_val_loss']
        df = pd.DataFrame(self.data, columns=columns)
        df.to_csv('history.csv', index=False)

def get_class_names(df):
    labels = df['class']
    return list(labels.unique())
