import pandas as pd
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
# import sys
# sys.path.append('../')
from Data_Visualization.eda import EDA_Visualization

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

# Forward fill
class FillMissingByLastKnownValue(FillMissingStrategy):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(method='ffill')

# Backward fill
class FillMissingByNextKnownValue(FillMissingStrategy):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(method='bfill')
    
class NotFill(FillMissingStrategy):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data


class DataCleansing:
    def __init__(self, data: pd.DataFrame = None):
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

        data = data if data is not None else self.data
        # drop duplicate rows
        data = data.drop_duplicates()
        # drop columns with same column names
        data = data.loc[:, ~data.columns.duplicated()]

        return data

    
    def handle_missing_values(self, data: pd.DataFrame = None, target_list: list = [], drop_threshold: float = 0.5, fill_missing_strategy=FillMissingStrategy) -> pd.DataFrame:
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
        
        data = data if data is not None else self.data
        # drop rows with missing values in the target_list
        data = data.dropna(subset=target_list)

        # drop columns with missing values more than the threshold
        dropped_cols = [i for i in data.columns if data[i].isnull().sum() / data.shape[0] > drop_threshold]
        print("Dropped columns:", dropped_cols)
        data = data.drop(columns=dropped_cols)

        # fill missing values in the remaining columns
        data = fill_missing_strategy.fill_missing(data)

        return data

    
    def detect_anomalies(self, column):
        # if self.data is not None:
        #     z_scores = np.abs(stats.zscore(self.data[column]))
        #     anomalies = self.data[np.abs(z_scores) > 3]
        #     return anomalies
        # else:
        #     print("No data loaded")
        pass
    
    def detect_outliers(self, data: pd.DataFrame = None, col_name: str = None , threshold : float = 3) -> pd.DataFrame:
        '''
        Detect and plot outliers for a column in the data

        Parameters:
        - data: input data
        - col_name: the column name to be checked
        - threshold: the threshold for detecting outliers using z-score
        '''

        data = data if data is not None else self.data

        z_scores = np.abs(stats.zscore(self.data[col_name]))
        outliers_index_list = np.where(z_scores > threshold)
        print("%d outliers detected" % len(outliers_index_list[0]))

        # get the lower and upper bound of the outliers
        # lower_bound = np.mean(data[col_name]) - threshold * np.std(data[col_name])
        # upper_bound = np.mean(data[col_name]) + threshold * np.std(data[col_name])
        # print("Lower bound: %.2f" % lower_bound)
        # print("Upper bound: %.2f" % upper_bound)

        eda_vis = EDA_Visualization()
        eda_vis.visualize_outliers(data, col_name, outliers_index_list)

        return
