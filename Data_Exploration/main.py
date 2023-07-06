import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt
import seaborn as sns

class DataExploration:
    def __init__(self):
        self.data = None
    
    def load_data(self, file_path):
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.data = pd.read_excel(file_path)
    
    def get_data_size(self):
        if self.data is not None:
            return self.data.shape
        else:
            return "No data loaded"
    
    def get_data_type(self):
        if self.data is not None:
            # return self.data.dtypes
            # get the number of each data type
            return self.data.dtypes.value_counts()
        else:
            return "No data loaded"
    
    def summarize_missing_data(self):
        if self.data is not None:
            missing_data = self.data.isnull().sum()
            return missing_data[missing_data > 0]
        else:
            return "No data loaded"
    
    def visualize_missing_data(self):
        if self.data is not None:
            if self.data.shape[0] > 50:
                print("Too many features to visualize")
                return
            msno.bar(self.data)
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

