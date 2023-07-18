import pandas as pd
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod

class FillMissingStrategy(ABC):
    @abstractmethod
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class FillMissingByMean(FillMissingStrategy):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(data.mean())

class FillMissingByMedian(FillMissingStrategy):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(data.median())

class FillMissingByMode(FillMissingStrategy):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(data.mode().iloc[0])

class FillMissingByLastKnownValue(FillMissingStrategy):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(method='ffill')
    
class NotFill(FillMissingStrategy):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data
    

class DataCleansing:
    def __init__(self, data):
        self.data = data
        # self.fill_missing_strategy = fill_missing_strategy
    
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Remove duplicates from the data

        Parameters:
        - data: the data to be cleaned

        Returns:
        - data: the data after removing duplicates
        '''
        data = data.drop_duplicates()
        return data

    
    def handle_missing_values(self, data: pd.DataFrame, target_list: list = [], drop_threshold: float = 0.5, fill_missing_strategy=FillMissingStrategy) -> pd.DataFrame:
        '''
        Handle missing values in the data

        Parameters:
        - data: the data to be cleaned
        - target_list: the list of columns to be used as target
        - drop_threshold: the threshold for dropping columns with missing values
        - fill_missing_strategy: the strategy for filling missing values

        Returns:
        - data: the data after handling missing values
        '''
         
        # drop rows with missing values in the target_list
        data = data.dropna(subset=target_list)

        # drop columns with missing values more than the threshold
        dropped_cols = [i for i in data.columns if data[i].isnull().sum() / data.shape[0] > drop_threshold]
        print("Dropped columns:", dropped_cols)
        data = data.drop(columns=dropped_cols)

        # fill missing values in the remaining columns
        data = fill_missing_strategy.fill_missing(data)
        # if strategy == 'mean':
        #     data = data.fillna(data.mean())
        # elif strategy == 'median':
        #     data = data.fillna(data.median())
        # elif strategy == 'mode':
        #     data = data.fillna(data.mode().iloc[0])
        # # fill missing values with the last known non-null value
        # elif strategy == 'ffill':
        #     data = data.fillna(method='ffill')
        # elif strategy == None:
        #     pass
        # else:
        #     print("Invalid strategy for handling missing values.")

        return data

    
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