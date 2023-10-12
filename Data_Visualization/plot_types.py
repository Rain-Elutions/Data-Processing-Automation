import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import List

def lineChart(data: pd.DataFrame, feat_list: List[str]):
        '''
        line chart for visualizing features in time series data

        Parameters:
        - data: the time series data
        - feat_list: the list of features to be visualized
        '''
        
        fig = px.line(data, x=data.index, y=feat_list, title=f'Time Series Plot of {feat_list}')
        fig.update_layout(
                width=1200,
                height=600,
                xaxis_title="Time",
                yaxis_title="Values",
        )
        fig.show()

        return

def barChart(x_list: list, y_list: list, title: str, x_label: str, y_label: str):
        fig = px.bar(x=x_list, y=y_list, title=title)
        fig.update_layout(
                xaxis_title=x_label,
                yaxis_title=y_label,
        )
        fig.show()
        return

def corrBar(data):
        fig = px.bar(data, y=data.index, color=data.values,
             color_continuous_scale='RdYlGn',
             labels={'x': 'Correlation', 'y': 'Features'},
             title='Correlation Plot')

        fig.update_layout(height=600, width=900)
        fig.show()
        return

def heatMap(data,settitle:str=''):
        fig = go.Figure(data=go.Heatmap(
                    z=data,
                    x=data.columns,
                    y=data.columns,
                    colorscale=px.colors.diverging.RdYlBu)
                    )
        fig.update_layout(
                title = settitle,
                title_x=0.5,
        )
        fig.update_scenes(aspectmode='data')
        fig.show()
        return

def single_column_vis(df: pd.DataFrame, col_name: str, title_name: str) -> go.Figure:
    """
    Helper function to visualize a single column

    Parameters
    ----------
    df : pd.DataFrame
        data of interest
    col_name : str
        column to visualize
    title_name : str
        title for the plot
    
    Returns
    -------
    fig : go.Figure
        visualization for a single variable
    """
    
    fig = make_subplots(rows=2, cols=2,
                        specs=[
                               [{'colspan':2}, None],
                               [{'type':'histogram'}, {'type':'bar'}]],
                        column_widths=[0.5,0.5],
                        vertical_spacing=0.1, horizontal_spacing=0.1,
                        subplot_titles=(
                                        f'Daily {col_name} Trend',
                                        f'{col_name} Distribution',
                                        f'{col_name} Box')
                       )

    # Top
    df_day = df.groupby(df.index.date)[col_name].mean()
    fig.add_trace(go.Scatter(x=df_day.index, y=df_day,
                             mode='lines',
                             marker=dict(color=px.colors.sequential.Viridis[5]),
                             name='Daily Trend'),
                  row=1, col=1)

    # Left Bottom Chart
    fig.add_trace(go.Histogram(x=df[col_name], 
                               name='Distribution', 
                               marker = dict(color = px.colors.sequential.Viridis[3])
                               ), 
                  row = 2, col = 1)

    fig.update_xaxes(showgrid = False, showline = True, 
                     linecolor = 'gray', linewidth = 2, 
                     row = 2, col = 1)
    fig.update_yaxes(showgrid = False, gridcolor = 'gray', 
                     gridwidth = 0.5, showline = True, 
                     linecolor = 'gray', linewidth = 2, 
                     row = 2, col = 1)

    # Right Bottom Chart
    fig.add_trace(go.Box(y=df[col_name],
                         name=f'{col_name}Box'

                             ),
                             row=2, col=2)

    fig.update_xaxes(showgrid = False, linecolor='gray', 
                     linewidth = 2, zeroline = False, 
                     row=2, col=2)
    fig.update_yaxes(showgrid = False, linecolor='gray',
                     linewidth=2, zeroline = False, 
                     row=2, col=2)


    # General Styling
    # NOTE: again consider doing this else where ...
    fig.update_layout(height=700, bargap=0.2,
                      margin=dict(b=50,r=30,l=100),
                      title = f"<span style='font-size:36px; font-family:Times New Roman'>{col_name} {title_name}</span>",                  
                      plot_bgcolor='rgb(242,242,242)',
                      paper_bgcolor = 'rgb(242,242,242)',
                      font=dict(family="Times New Roman", size= 14),
                      hoverlabel=dict(font_color="floralwhite"),
                      showlegend=False)
    return fig

def two_line_plot(df: pd.DataFrame, col_one: str, col_two: str) -> go.Figure:
    """
    Function to make two line plots

    Parameters
    ----------
    df : pd.DataFrame
        data of interest
    col_one : str
        name of the first variable of interest
    col_two : str
        name of the second variable of interest

    Returns
    -------
    fig : go.Figure
        two line plots
    """
    
    fig = make_subplots(rows=1, cols=2,
                        specs=[
                               [{'colspan':2}, None]
                               ],

                        vertical_spacing=0.1, horizontal_spacing=0.1,
                        subplot_titles=(
                                        'Daily Trend',
                                        )
                       )


    # Top Chart
    df_day = df.groupby(df.index.date)[col_one].mean()
    fig.add_trace(go.Scatter(x=df_day.index, y=df_day,
                             mode='lines',
                             marker=dict(color=px.colors.sequential.Viridis[2]),
                             name=f'{col_one}'),
                  row=1, col=1)

    df_day = df.groupby(df.index.date)[col_two].mean()
    fig.add_trace(go.Scatter(x=df_day.index, y=df_day,
                             mode='lines',
                             marker=dict(color=px.colors.sequential.Viridis[8]),
                             name=f'{col_two}'),
                  row=1, col=1)

    # General Styling
    # NOTE: could also do this elsewhere
    fig.update_layout(height=800, bargap=0.2,
                      margin=dict(b=50,r=30,l=100),
                      title = "<span style='font-size:36px; font-family:Times New Roman'>Input & Output</span>",                  
                      plot_bgcolor='rgb(242,242,242)',
                      paper_bgcolor = 'rgb(242,242,242)',
                      font=dict(family="Times New Roman", size= 14),
                      hoverlabel=dict(font_color="floralwhite"),
                      showlegend=True)
    return fig


class BoxPlots:
        def __init__(self,df1: pd.DataFrame,df2: pd.DataFrame,target_name: str = '', col_name1 = '',col_name2=''):
                self.df1 = df1
                self.df2 = df2
                self.target_name = target_name
                self.col_name1 = col_name1
                self.col_name2 = col_name2


        def double_boxplot(self) -> go.Figure:
                '''
                Function to make side by side boxplots for 
                the top 10 attributes, based on whether the 
                output was optimal or not
                ----------
                df1 : pd.DataFrame
                        data of optimal bathces 
                df2 : pd.DataFrame
                        data of suboptimal bathces 
                col_name1: str
                        First tag of interest 
                col_name2: str
                        Second tag of interest 
                Returns
                -------
                plot: go.Figure
                        plotly go figure
                '''
                #Making the first Subplot
                #adding the first box
                fig = make_subplots(rows=1, cols=2,
                                        specs=[
                                        [{'type':'bar'}, {'type':'bar'}]
                                        ],
                                        column_widths=[0.5,0.5],
                                        vertical_spacing=0.1, horizontal_spacing=0.1,
                                        subplot_titles=(
                                                        self.col_name1,
                                                        self.col_name2
                                                        )
                                )
                fig.add_trace(go.Box(y=self.df1[self.col_name1],
                                        name='optimal'

                                        ),
                                        row=1, col=1)


                #adding the q1,q2,q3 and mean to plot
                fig.add_annotation(x=0.4, y=np.nanpercentile(self.df1[self.col_name1],25)-(abs(np.nanpercentile(self.df1[self.col_name1],25))*.05), 
                        text="25%: " + str(np.nanpercentile(self.df1[self.col_name1],25).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        )
                fig.add_annotation(x=0.4, y=np.nanpercentile(self.df1[self.col_name1],50), #Median
                        text="Median: " + str(np.nanpercentile(self.df1[self.col_name1],50).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        )
                fig.add_annotation(x=0.4, y=np.nanpercentile(self.df1[self.col_name1],75)+(abs(np.nanpercentile(self.df1[self.col_name1],75))*.05), 
                        text="75%: " + str(np.nanpercentile(self.df1[self.col_name1],75).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        )
                mean_value = np.mean(self.df1[self.col_name1])
                if not np.isnan(np.mean(self.df1[self.col_name1])):
                        text = "Mean: " + str(mean_value.round(2))
                else:
                        text = "Mean: N/A"
                fig.add_annotation(x=0.4, y=np.min(self.df1[self.col_name1])-((abs(np.min(self.df1[self.col_name1]))*.15)+1), 
                        text=text,
                        font=dict(size=12),
                        showarrow=False,
                        )       

                #adding the second box
                fig.add_trace(go.Box(y=self.df2[self.col_name1],
                                        name='suboptimal'

                                        ),
                                        row=1, col=1)


                #adding the q1,q2,q3 and mean to plot
                fig.add_annotation(x=1.4, y=np.nanpercentile(self.df2[self.col_name1],25)-(abs(np.nanpercentile(self.df2[self.col_name1],25))*.05), 
                        text="25%: " + str(np.nanpercentile(self.df2[self.col_name1],25).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        )
                fig.add_annotation(x=1.4, y=np.nanpercentile(self.df2[self.col_name1],50), #Median
                        text="Median: " + str(np.nanpercentile(self.df2[self.col_name1],50).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        )
                fig.add_annotation(x=1.4, y=np.nanpercentile(self.df2[self.col_name1],75)+(abs(np.nanpercentile(self.df2[self.col_name1],75))*.05), 
                        text="75%: " + str(np.nanpercentile(self.df2[self.col_name1],75).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        )
                mean_value = np.mean(self.df2[self.col_name1])
                if not np.isnan(np.mean(self.df2[self.col_name1])):
                        text = "Mean: " + str(mean_value.round(2))
                else:
                        text = "Mean: N/A"
                fig.add_annotation(x=1.4, y=np.min(self.df2[self.col_name1])-((abs(np.min(self.df2[self.col_name1]))*.15)+1), 
                        text=text,
                        font=dict(size=12),
                        showarrow=False,
                        )

                #adding the axes
                fig.update_xaxes(showgrid = False, linecolor='gray', 
                                linewidth = 2, zeroline = False, 
                                row=1, col=1)
                fig.update_yaxes(showgrid = False, linecolor='gray',
                                linewidth=2, zeroline = False, 
                                row=1, col=1)


                #Making the second Subplot
                #adding the first box
                fig.add_trace(go.Box(y=self.df1[self.col_name2],
                                        name='optimal'

                                        ),
                                        row=1, col=2)
                
                #adding the q1,q2,q3 and mean to plot
                fig.add_annotation(x=0.4, y=np.nanpercentile(self.df1[self.col_name2],25)-(abs(np.nanpercentile(self.df1[self.col_name2],25))*.05), 
                        text="25%: " + str(np.nanpercentile(self.df1[self.col_name2],25).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        row=1, col=2
                        )
                fig.add_annotation(x=0.4, y=np.nanpercentile(self.df1[self.col_name2],50), #Median
                        text="Median: " + str(np.nanpercentile(self.df1[self.col_name2],50).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        row=1, col=2
                        )
                fig.add_annotation(x=0.4, y=np.nanpercentile(self.df1[self.col_name2],75)+(abs(np.nanpercentile(self.df1[self.col_name2],75))*.05), 
                        text="75%: " + str(np.nanpercentile(self.df1[self.col_name2],75).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        row=1, col=2
                        )
                mean_value = np.mean(self.df1[self.col_name2])
                if not np.isnan(np.mean(self.df1[self.col_name2])):
                        text = "Mean: " + str(mean_value.round(2))
                else:
                        text = "Mean: N/A"
                fig.add_annotation(x=0.4, y=np.min(self.df1[self.col_name2])-((abs(np.min(self.df1[self.col_name2])*.15))+1), 
                        text=text,
                        font=dict(size=12),
                        showarrow=False,
                        row=1, col=2
                        )       

                #adding the second box
                fig.add_trace(go.Box(y=self.df2[self.col_name2],
                                        name='suboptimal'

                                        ),
                                        row=1, col=2)

                #adding the q1,q2,q3 and mean to plot
                fig.add_annotation(x=1.4, y=np.nanpercentile(self.df2[self.col_name2],25)-(abs(np.nanpercentile(self.df2[self.col_name2],25))*.05), 
                        text="25%: " + str(np.nanpercentile(self.df2[self.col_name2],25).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        row=1, col=2
                        )
                fig.add_annotation(x=1.4, y=np.nanpercentile(self.df2[self.col_name2],50), #Median
                        text="Median: " + str(np.nanpercentile(self.df2[self.col_name2],50).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        row=1, col=2
                        )
                fig.add_annotation(x=1.4, y=np.nanpercentile(self.df2[self.col_name2],75)+(abs(np.nanpercentile(self.df2[self.col_name2],75))*.05), 
                        text="75%: " + str(np.nanpercentile(self.df2[self.col_name2],75).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        row=1, col=2
                        )
                mean_value = np.mean(self.df2[self.col_name2])
                if not np.isnan(np.mean(self.df2[self.col_name2])):
                        text = "Mean: " + str(mean_value.round(2))
                else:
                        text = "Mean: N/A"
                fig.add_annotation(x=1.4, y=np.min(self.df2[self.col_name2])-((abs(np.min(self.df2[self.col_name2]))*.15)+1), 
                        text=text,
                        font=dict(size=12),
                        showarrow=False,
                        row=1, col=2
                        )

                #adding axes
                fig.update_xaxes(showgrid = False, linecolor='gray', 
                                linewidth = 2, zeroline = False, 
                                row=1, col=2)
                fig.update_yaxes(showgrid = False, linecolor='gray',
                                linewidth=2, zeroline = False, 
                                row=1, col=2)

                #General Formatting 
                fig.update_layout(height=700, bargap=0.2,
                                        margin=dict(b=50,r=30,l=100),                
                                        plot_bgcolor='rgb(242,242,242)',
                                        paper_bgcolor = 'rgb(242,242,242)',
                                        font=dict(family="Times New Roman", size= 14),
                                        hoverlabel=dict(font_color="floralwhite"),
                                        showlegend=True)

                if ":" in self.col_name1:
                        self.col_name1 = self.col_name1.replace(":","_")
                if ":" in self.col_name2:
                        self.col_name2 = self.col_name2.replace(":","_")
                plotly.io.write_image(fig, file= './Data_Cleansing/'+ self.target_name +'_anomaly_report/graphics/' + self.col_name1 + self.col_name2 + '.png',format='png', width=1600,height=950)
                
                return