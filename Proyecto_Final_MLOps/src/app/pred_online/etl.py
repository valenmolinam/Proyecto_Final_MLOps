import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

class Dataloader:
    def __init__(self, path_to_save, n_samples=100000):
        self.df = None
        self.path_to_save = path_to_save
        self.n_samples = n_samples

    def download_data(self):
        path = kagglehub.dataset_download("ealaxi/paysim1", self.path_to_save)
        return path
    
    def load_data(self):
        if self.path_to_save:
            self.df = pd.read_csv(self.path_to_save, nrows=self.n_samples)
        else:
            downloaded_path = self.download_data()
            self.df = pd.read_csv(downloaded_path, nrows=self.n_samples)
        return self.df
    
    def drop_name_columns(self):
        if self.df is not None:
            self.df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)