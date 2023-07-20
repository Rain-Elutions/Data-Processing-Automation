import pandas as pd
import plotly.graph_objects as go

class EDA_Visualization:
    def __init__(self, data):
        self.data = data

    def visualize_missing_data(self, data: pd.DataFrame = None):
        """
        Visualize the missing data

        Parameters:
        - data: the input data

        Returns:
        - the missing data plots
        """

        data = data if data is not None else self.data
    
        if self.data.shape[1] > 200:
            print("Too many features to visualize")
            return
        num_cols_one_plot = 40
        num_plots = self.data.shape[1] // num_cols_one_plot + 1
        for i in range(num_plots):
            start = i * num_cols_one_plot
            end = min((i+1) * num_cols_one_plot, self.data.shape[1])
            missing_counts = self.data.iloc[:, start:end].isnull().sum()
            # Create a bar plot of missing values using Plotly
            fig = go.Figure(data=[go.Bar(x=missing_counts.index, y=missing_counts)])
            fig.update_layout(
                title="Missing Data",
                xaxis_title="Columns",
                yaxis_title="Missing Count",
            )
            fig.update_yaxes(
                rangemode = "tozero"
            )
            fig.show()
            
        return
