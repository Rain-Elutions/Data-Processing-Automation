import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Visualization.plot_types import barChart, corrBar, heatMap
from Data_Analyzing.correlation_analysis import CorrelationTypes

class DataAnalysis:
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        self.data = data
        self.target_name = target_name

    def correlation_analysis(self, thresh: float = 0):
        '''
        Calculate the correlations of each column for linear and non linear relations

        Parameters:
        -data: the input Dataframe
        -thresh: the threshold for correlations to include in the heatmap
        
        '''
        # getting linear full correlations
        p = CorrelationTypes(self.data,10,self.target_name)
        corrplot = p.get_correlations()

        # getting nonlinear correlations
        spearman = p.non_linear()

        # creating heatmaps
        heatMap(corrplot.iloc[:,:10],'Correlation')
        heatMap(spearman.iloc[:,:10],'Spearman Correlation')
        # if isinstance(MIaug,pd.DataFrame):
        #     heatMap(MIaug.iloc[:,:10],'Mutual Information')
        # return 

    def variance_analysis(self, data: pd.DataFrame = None):
        '''
        Calculate the Normalized STD of each column, as a measure of variance

        Parameters:
        - data: the input DataFrame

        '''

        data = self.data if data is None else data

        norm_std = data.std() / abs(data.mean() + 1e-6)
        norm_std = norm_std.sort_values(ascending=False)
        # visualize
        barChart(x_list=norm_std.index, y_list=norm_std.values, title="Norm STD of each column", x_label="Column Name", y_label="Norm STD")

        return
    
    def feature_selection(self):
        pass
