import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Visualization.plot_types import barChart,corrBar,heatMap
from correlation_report import get_correlations

class DataAnalysis:
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        self.data = data
        self.target_name = target_name
    def correlation_analysis(self,thresh:float = 0.5):
        corrplot = get_correlations(self.data)
        # corrplot.to_xlsx('./Data_Analyzing/')  #Place Holder
        corrBar(corrplot)
        heatMap(corrplot,thresh)

        return corrplot

    def variance_analysis(self, data: pd.DataFrame = None):
        '''
        Calculate the Normalized STD of each column, as a measure of variance

        Parameters:
        - data: the input DataFrame

        '''

        data = self.data if data is None else data

        norm_std = self.data.std() / abs(self.data.mean() + 1e-6)
        norm_std = norm_std.sort_values(ascending=False)
        # visualize
        barChart(x=norm_std.index, y=norm_std.values, title="Norm STD of each column", x_label="Column Name", y_label="Norm STD")

        return
    
    def feature_selection(self):
        pass
