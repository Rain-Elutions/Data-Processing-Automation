import pandas as pd
import numpy as np
from scipy import stats

class DataCleansing:
    def __init__(self, data):
        self.data = data
    
    def remove_duplicates(self):
        if self.data is not None:
            self.data.drop_duplicates(inplace=True)
        else:
            print("No data loaded")
    
    def handle_missing_values(self, strategy='mean'):
        # if self.data is not None:
        #     if strategy == 'mean':
        #         self.data.fillna(self.data.mean(), inplace=True)
        #     elif strategy == 'median':
        #         self.data.fillna(self.data.median(), inplace=True)
        #     elif strategy == 'mode':
        #         self.data.fillna(self.data.mode().iloc[0], inplace=True)
        #     else:
        #         print("Invalid strategy for handling missing values.")
        # else:
        #     print("No data loaded")
        pass
    
    def detect_anomalies(self, column):
        # if self.data is not None:
        #     z_scores = np.abs(stats.zscore(self.data[column]))
        #     anomalies = self.data[np.abs(z_scores) > 3]
        #     return anomalies
        # else:
        #     print("No data loaded")
        pass
    
    def remove_outliers(self, column):
        if self.data is not None:
            z_scores = np.abs(stats.zscore(self.data[column]))
            self.data = self.data[np.abs(z_scores) <= 3]
        else:
            print("No data loaded")