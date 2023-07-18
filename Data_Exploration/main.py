import pandas as pd
import plotly.graph_objects as go
# import missingno as msno
# from matplotlib import pyplot as plt


class DataExploration:
    def __init__(self):
        self.data = None
    
    def load_data(self, file_path, parse_dates=True, index_col=0) -> pd.DataFrame:
        """
        Load data into memory
        
        Parameters:
        - file_path: the path of the data file
        - parse_dates: whether to parse the dates
        - index_col: the index column

        Returns:
        - self.data: the raw data
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, parse_dates=parse_dates, index_col=index_col)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path, parse_dates=parse_dates, index_col=index_col)
            return self.data
        except Exception as e:
            print("Error:", str(e))
    
    
    def get_data_size(self, data: pd.DataFrame = None) -> tuple:
        """
        Get the size of the data

        Parameters:
        - data: the input data
        
        Returns:
        - self.data.shape: the size of the data
        """
        data = data if data is not None else self.data
        return data.shape


    def get_data_type(self, data: pd.DataFrame = None, column_name=None) -> str:
        """
        Return the data type for the specified column

        Parameters:
        - data: the input data
        - column_name: the name of the column
        
        Returns:
        - the data type for the specified column
        """
        try:
            data = data if data is not None else self.data
            return data.dtypes[column_name]
        except Exception as e:
            print("Invalid column name")
            print("Error:", str(e))
    
    def summarize_data_type(self, data: pd.DataFrame = None):
        """
        Return the data types and the number of each type for all data

        Parameters:
        - data: the input data
        
        Returns:
        - the number of each data type
        """

        data = data if data is not None else self.data
        # return self.data.dtypes
        return data.dtypes.value_counts()

    
    def summarize_missing_data(self, data: pd.DataFrame = None) -> pd.Series:
        """
        Summarize the number of missing data

        Parameters:
        - data: the input data

        Returns:
        - the number of missing data for columns with missing data
        """

        data = data if data is not None else self.data
        missing_data = data.isnull().sum()
        return missing_data[missing_data > 0]