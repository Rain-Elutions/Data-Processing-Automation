import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from matplotlib import pyplot as plt

class EDA_Visualization:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data

    def visualize_missing_data(self, data: pd.DataFrame = None):
        """
        Visualize the missing data

        Parameters:
        - data: the input data

        """

        data = data if data is not None else self.data
    
        if data.shape[1] > 200:
            print("Too many features to visualize")
            return
        num_cols_one_plot = 40
        num_plots = data.shape[1] // num_cols_one_plot + 1
        for i in range(num_plots):
            start = i * num_cols_one_plot
            end = min((i+1) * num_cols_one_plot, data.shape[1])
            missing_counts = data.iloc[:, start:end].isnull().sum()
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
    
    def visualize_outliers(self, data: pd.DataFrame = None, col_name: str = None, outlier_index_list: list = None):
        """
        Visualize the outliers

        Parameters:
        - data: the input data
        - col_name: the name of the column to be visualized
        - outlier_index_list: the list of indices of outliers (not timestamps)

        """
        data = data if data is not None else self.data

        fig = px.line(data, x=data.index, y=col_name, title='Outliers for %s' % col_name)
        fig.add_trace(go.Scatter(x=data.index[outlier_index_list], y=data[col_name].values[outlier_index_list], mode='markers', name='outliers'))
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Value",
        )
        fig.show()

        return

    def visualize_feature_importance(self, imp_dict: dict, highlight_feats: list[str]=[]):
        '''
        Visualize the feature importance

        Parameters:
        - imp_dict: the dictionary of feature importance
        - highlight_feats: the list of features to be highlighted in the plot
        '''

        # plot imp_dict in the sorted order
        cv_importances = np.array(list(imp_dict.values()))
        plt.figure(figsize=(10, 15))
        plt.barh(np.array(list(imp_dict.keys()))[np.argsort(cv_importances)], cv_importances[np.argsort(cv_importances)])

        # Add title and axis names
        plt.title('Feature Importances')
        plt.xlabel('Average Scaled Feature importances')
        plt.ylabel('Features')

        # Highlight labels
        [t.set_color('red') for t in plt.gca().get_yticklabels() if t.get_text() in highlight_feats]

        # Add y values
        for i, v in enumerate(cv_importances[np.argsort(cv_importances)]):
            plt.text(v, i, " "+str(round(v, 2)), color='blue', va='center')

        plt.show()

        return
    
    # need test
    def visualize_correlation(self, data: pd.DataFrame = None, target_name: str = '', highlight_feats: list[str] = []):
        '''
        Visualize the correlation between features and target

        Parameters:
        - data: the input data
        - corr_type: the type of correlation to be calculated

        '''

        data = data if data is not None else self.data
        if target_name == '':
            raise ValueError('target_name cannot be empty')
        
        corr = data.corr()[target_name].sort_values(ascending=False)
        corr = corr.reset_index().rename(columns={'index': 'Features', target_name: 'Correlation'})
        corr['Correlation'] = corr['Correlation'].round(2)
        corr = corr[corr['Features'] != target_name]

        fig = px.bar(corr, x='Features', y='Correlation', color='Correlation', color_continuous_scale='RdBu')
        fig.update_layout(title='Correlation Between the Target Variable and Features')
        fig.update_layout(width=1200)
        fig.update_xaxes(tickfont=dict(size=8))

        # highlight the colored tags in xticks
        fig.update_xaxes(tickmode='array', tickvals=corr['Features'], ticktext=['<b>' + col + '</b>' if col in highlight_feats else col for col in corr['Features']])
        fig.update_yaxes(range=[-1, 1])

        fig.show()
