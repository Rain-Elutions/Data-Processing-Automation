import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class BoxPlots:
        def __init__(self,df1: pd.DataFrame,df2 = pd.DataFrame(), col_name1 = '',col_name2=''):
                self.df1 = df1
                self.df2 = df2
                self.col_name1 = col_name1
                self.col_name2 = col_name2


        def double_boxplot(self) -> go.Figure:
                '''
                Function to make side by side boxplots for 
                the top 10 attributes, based on whether the 
                output was good or not
                ----------
                df1 : pd.DataFrame
                        data of good bathces 
                df2 : pd.DataFrame
                        data of bad bathces 
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
                                        name='Good'

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

                fig.add_annotation(x=0.4, y=np.min(self.df1[self.col_name1])-((abs(np.min(self.df1[self.col_name1]))*.15)+1), 
                        text="Mean: " + str(np.mean(self.df1[self.col_name1]).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        )       

                #adding the second box
                fig.add_trace(go.Box(y=self.df2[self.col_name1],
                                        name='Bad'

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
                fig.add_annotation(x=1.4, y=np.min(self.df2[self.col_name1])-((abs(np.min(self.df2[self.col_name1]))*.15)+1), 
                        text="Mean: " + str(np.mean(self.df2[self.col_name1]).round(2)),
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
                                        name='Good'

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

                fig.add_annotation(x=0.4, y=np.min(self.df1[self.col_name2])-((abs(np.min(self.df1[self.col_name2])*.15))+1), 
                        text="Mean: " + str(np.mean(self.df1[self.col_name2]).round(2)),
                        font=dict(size=12),
                        showarrow=False,
                        row=1, col=2
                        )       

                #adding the second box
                fig.add_trace(go.Box(y=self.df2[self.col_name2],
                                        name='Bad'

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
                fig.add_annotation(x=1.4, y=np.min(self.df2[self.col_name2])-((abs(np.min(self.df2[self.col_name2]))*.15)+1), 
                        text="Mean: " + str(np.mean(self.df2[self.col_name2]).round(2)),
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
                plotly.io.write_image(fig, file= './anomaly_report/graphics/' + self.col_name1 + self.col_name2 + '.png',format='png', width=1600,height=950)
                plotly.io.write_image(fig, file= './anomaly_report/graphics/' + self.col_name1 + self.col_name2 + '.pdf',format='pdf', width=1600,height=950)
                
                
                
                return