import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt
import plotly.graph_objects as go

class DataExploration:
    def __init__(self):
        self.data = None
    
    def load_data(self, file_path, parse_dates=True, index_col=0):
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
    
    
    def get_data_size(self):
        """
        Get the size of the data
        
        Returns:
        - self.data.shape: the size of the data
        """
        try:
            return self.data.shape
        except Exception as e:
            print("Error:", str(e))
    
    def get_data_type(self):
        """
        Return the data types and the number of each type
        
        Returns:
        - the number of each data type
        """
        try:
            # return self.data.dtypes
            return self.data.dtypes.value_counts()
        except Exception as e:
            print("Error:", str(e))
    
    def summarize_missing_data(self):
        """
        Summarize the number of missing data

        Returns:
        - the number of missing data for columns with missing data
        """
        try:
            missing_data = self.data.isnull().sum()
            return missing_data[missing_data > 0]
        except Exception as e:
            print("Error:", str(e))
    
    def visualize_missing_data(self):
        """
        Visualize the missing data
        
        Returns:
        - the missing data plots
        """
        try:
            if self.data.shape[1] > 200:
                print("Too many features to visualize")
                return
            num_cols_one_plot = 40
            num_plots = self.data.shape[1] // num_cols_one_plot + 1
            for i in range(num_plots):
                start = i * num_cols_one_plot
                end = min((i+1) * num_cols_one_plot, self.data.shape[1])
                # msno.bar(self.data.iloc[:, start:end])
                # plt.show()

                missing_counts = self.data.iloc[:, start:end].isnull().sum()
                # Create a bar plot of missing values using Plotly
                fig = go.Figure(data=[go.Bar(x=missing_counts.index, y=missing_counts)])
                fig.update_layout(
                    title="Missing Data",
                    xaxis_title="Columns",
                    yaxis_title="Missing Count",
                )
                fig.show()
            return
        except Exception as e:
            print("Error:", str(e))
