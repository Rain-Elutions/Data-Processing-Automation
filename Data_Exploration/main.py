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

        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path, parse_dates=parse_dates, index_col=index_col)
        elif file_path.endswith('.xlsx'):
            self.data = pd.read_excel(file_path, parse_dates=parse_dates, index_col=index_col)
        elif file_path.endswith('.pickle'):
            self.data = pd.read_pickle(file_path)
            self.data = self.data.set_index(self.data.columns[0])
            if parse_dates:
                self.data.index = pd.to_datetime(self.data.index)

        return self.data
    
    
    def get_data_size(self, data: pd.DataFrame = None):
        """
        Get the size of the data

        Parameters:
        - data: the input data
        """

        data = data if data is not None else self.data
        print("Data size:", data.shape)

        return


    def get_data_type(self, data: pd.DataFrame = None, column_name: str = None):
        """
        Return the data type for the specified column

        Parameters:
        - data: the input data
        - column_name: the name of the column
        """

        data = data if data is not None else self.data
        print("Data type of column", column_name, "is: ", data[column_name].dtype)

        return

    
    def summarize_data_type(self, data: pd.DataFrame = None):
        """
        Return the data types and the number of each type for all data

        Parameters:
        - data: the input data
        """

        data = data if data is not None else self.data
        print("Data types summary:\n", data.dtypes.value_counts())

        return

    
    def summarize_missing_data(self, data: pd.DataFrame = None):
        """
        Summarize the number of missing data

        Parameters:
        - data: the input data
        """

        data = data if data is not None else self.data
        missing_data = data.isnull().sum()
        print("Missing data summary:")
        print(missing_data[missing_data > 0])

        return