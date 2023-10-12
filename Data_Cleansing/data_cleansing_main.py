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
import json


# Strategy Pattern using ABC
class FillMissingMethod(ABC):
    @abstractmethod
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class FillMissingByMean(FillMissingMethod):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(data.mean())

class FillMissingByMedian(FillMissingMethod):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(data.median())

class FillMissingByMode(FillMissingMethod):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(data.mode().iloc[0])

# Forward fill
class FillMissingByLastKnownValue(FillMissingMethod):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(method='ffill')

# Backward fill
class FillMissingByNextKnownValue(FillMissingMethod):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(method='bfill')
    
# Interpolation Fill
class FillMissingByInterpolation(FillMissingMethod):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.interpolate(method='linear', limit_direction='both')
    
class NotFill(FillMissingMethod):
    def fill_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

def fill_method_factory(method="forward"):
    """Factory Method"""
    filler = {
        "mean": FillMissingByMean(),
        "median": FillMissingByMedian(),
        "mode": FillMissingByMode(),
        "forward": FillMissingByLastKnownValue(),
        "back": FillMissingByNextKnownValue(),
        "notfill": NotFill()
    }

    return filler[method]


class DataCleansing:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data
    
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

        # drop columns with all values be the same
        # duplicate_columns = data.columns[data.T.duplicated(keep='first')]
        # print("Dropped columns:", duplicate_columns)
        # data = data.drop(duplicate_columns, axis=1)

        # for columns with same column names, set an alert and keep the first one
        if len(data.columns) != len(set(data.columns)):
            print("Warning: There are columns with same column names, we have kept the first one and drop the rest")
            print("Dropped columns:", data.columns[data.columns.duplicated()])
            data = data.loc[:, ~data.columns.duplicated()]
        
        return data

    
    def handle_missing_values(self, data: pd.DataFrame = None, target_list: list = [], drop_thresh: float = 0.5, fill_missing_method="forward") -> pd.DataFrame:
        '''
        Handle missing values in the data

        Parameters:
        - data: the data to be cleaned
        - target_list: the list of columns to be used as target
        - drop_threshold: the threshold for dropping columns with missing values
        - fill_missing_method: the method for filling missing values

        Returns:
        - data: the data after handling missing values
        '''
        
        data = data if data is not None else self.data
        # convert "" to NaN
        data = data.replace(r'^\s*$', np.nan, regex=True)

        # drop rows with missing values in the target_list
        print("# rows dropped with missing values in the target variable:", data.shape[0] - data.dropna(subset=target_list).shape[0])
        data = data.dropna(subset=target_list)


        # drop columns that have missing values more than the threshold
        dropped_cols = [i for i in data.columns if data[i].isnull().sum() / data.shape[0] > drop_thresh]
        print("Dropped columns:", dropped_cols)
        data = data.drop(columns=dropped_cols)
        
        # fill missing values for the rest of the columns
        try:
            numeric_columns = data.select_dtypes(include=['number'])
            data[numeric_columns.columns] = fill_method_factory(fill_missing_method).fill_missing(numeric_columns)
            print("Filled missing values using %s" % fill_missing_method)
        except KeyError:
            print("Filling failed. Invalid fill missing method, choose from: mean, median, model, forward, back, notfill")
            
        return data

    def detect_shutdown(self, data: pd.DataFrame = None, manual_shutdown_thresh: float = None, drop_thresh: float = 0.5) -> pd.DataFrame:
        '''
        Detect shutdown period in the time-series data

        Parameters:
        - data: the data to be cleaned
        - manual_shutdown_thresh: the manually input threshold for detecting as a shutdown, such as 0
        - drop_thresh: the threshold for dropping columns, if the % of shutdown period is more than this threshold, the column will be dropped

        Returns:
        - data: the data after detecting shut down
        '''

        data = data if data is not None else self.data

        # get the threshold for shutdown period
        if manual_shutdown_thresh is None:
            thresh_list = []
            for col_name in data.columns:
                # using z-score to get the lower bound as the threshold
                thresh_list.append(np.mean(data[col_name]) - 3 * np.std(data[col_name]))
        else:
            thresh_list = [manual_shutdown_thresh] * len(data.columns)

        # save the threshold to a json file
        with open('temp_save/shutdown_thresh.json', 'w') as f:
            json.dump(thresh_list, f)

        # get the % of shutdown period for each column
        shutdown_percent = data[data <= thresh_list].count() / data.shape[0]
        # get the columns that have shutdown period more than the threshold
        dropped_cols = shutdown_percent[shutdown_percent > drop_thresh].index.tolist()
        print("Dropped columns:", dropped_cols)
        data = data.drop(columns=dropped_cols)

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
        Detect and plot outliers for a column in the data based on z-score
        z-score is a measure of how many standard deviations a data point is away from the mean

        Parameters:
        - data: the input data
        - col_name: the column name to be checked
        - threshold: the threshold for detecting outliers using z-score
        '''

        data = data if data is not None else self.data
        # Check missing values
        if data[col_name].isnull().sum() > 0:
            print("Missing values detected in %s" % col_name, ", please handle missing values first")
            return

        z_scores = np.abs(stats.zscore(data[col_name]))
        z_scores_dict = [{"index": i, "z_score": z} for i, z in enumerate(z_scores)]
        # save the threshold to a json file
        with open('./temp_save/z_scores.json', 'w') as f:
            json.dump(z_scores_dict, f)
        
        outliers_index_list = np.where(z_scores > threshold)
        if len(outliers_index_list[0]) > 0:
            print("%d outliers detected in " % len(outliers_index_list[0]) + col_name)

            # get the lower and upper bound of the outliers
            # lower_bound = np.mean(data[col_name]) - threshold * np.std(data[col_name])
            # upper_bound = np.mean(data[col_name]) + threshold * np.std(data[col_name])
            # print("Lower bound: %.2f" % lower_bound)
            # print("Upper bound: %.2f" % upper_bound)
        
            eda_vis = EDA_Visualization()
            eda_vis.visualize_outliers(data, col_name, outliers_index_list)

        return


# df = pd.read_csv('data/Essar_RE_Boilers_B21_sample.csv', parse_dates=True, index_col=0)
# df = pd.read_csv('data/sasol_data_sample.csv', parse_dates=True, index_col=0)
# generate a df with 50% of values 0
# df = pd.DataFrame(np.random.randint(0, 2, size=(100, 4)), columns=list('ABCD'))
# print(df)
# data_cleansing = DataCleansing(df)
# df = data_cleansing.handle_missing_values(df, target_list=[], drop_threshold=0.5, fill_missing_method=FillMissingByMean())
# data_cleansing.detect_outliers(df, col_name=df.columns[0], threshold=2.5)
# df = data_cleansing.detect_shutdown(df)