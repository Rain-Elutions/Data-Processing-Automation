import pandas as pd
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod
import sys
import os
# in python script, use absolute path of this file to add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Visualization.eda import EDA_Visualization
from Data_Cleansing.anomaly_detection import AnomalyDetection

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

    
    def generate_anomaly_report(self, data: pd.DataFrame = None, target_name : str = '', problem_type : str = 'max', manual_input=None, manual_thresh=None):
        '''
        Generate anomaly report for the target variable in the input data

        Parameters:
        - data: the input data
        - target_col: the column name of the target variable

        Returns:
        A anomaly_report folder that has:
        - graphics: the folder containing the plots
        - xlsx: the folder containing the excel file, including the correlation table and the final report
        - stats.txt: the file containing the basic stats for IES
        '''

        data = data if data is not None else self.data
        anom_detect = AnomalyDetection(data, target_name, problem_type=problem_type, manual_input=manual_input, manual_thresh=manual_thresh)
        anom_detect.anomaly_report()

        return
    
    def detect_outliers(self, data: pd.DataFrame = None, col_name: str = None , threshold : float = 3):
        '''
        Detect and plot outliers for a column in the data

        Parameters:
        - data: the input data
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
