import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt
import seaborn as sns

class DataExploration:
    def __init__(self):
        self.data = None
    
    def load_data(self, file_path):
        """
        Load data into memory
        
        Parameters:
        - file_path: the path of the data file
        """
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.data = pd.read_excel(file_path)
    
    def get_data_size(self):
        """
        Get the size of the data
        
        Returns:
        - self.data.shape: the size of the data
        """
        if self.data is not None:
            return self.data.shape
        else:
            return "No data loaded"
    
    def get_data_type(self):
        """
        Return the data types and the number of each type
        
        Returns:
        - the number of each data type
        """
        if self.data is not None:
            # return self.data.dtypes
            return self.data.dtypes.value_counts()
        else:
            return "No data loaded"
    
    def summarize_missing_data(self):
        """
        Summarize the number of missing data

        Returns:
        - the number of missing data for columns with missing data
        """
        if self.data is not None:
            missing_data = self.data.isnull().sum()
            return missing_data[missing_data > 0]
        else:
            return "No data loaded"
    
    def visualize_missing_data(self):
        """
        Visualize the missing data
        
        Returns:
        - the missing data plots
        """
        if self.data is not None:
            if self.data.shape[1] > 200:
                print("Too many features to visualize")
                return
            num_cols_one_plot = 40
            num_plots = self.data.shape[1] // num_cols_one_plot + 1
            for i in range(num_plots):
                start = i * num_cols_one_plot
                end = min((i+1) * num_cols_one_plot, self.data.shape[1])
                msno.bar(self.data.iloc[:, start:end])
                plt.show()
            return
        else:
            return "No data loaded"
    
    # def visualize_missing(self):
    #     sns.set(rc={'figure.figsize':(10,7)})
    #     sns.heatmap(self.data.isna(), yticklabels = False, cbar = False, cmap = plt.cm.magma)
    #     plt.title(label = 'Heatmap for Missing Values', fontsize = 16, color='red')
    #     plt.xlabel(xlabel = 'Features', fontsize = 16, color='red')
    #     plt.show() 

