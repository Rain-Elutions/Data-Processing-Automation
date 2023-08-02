import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from Data_Visualization.boxplots import *
from Data_Analyzing.correlation_report import *
from Data_Analyzing.feature_selection import * 

class AnomalyDetection:
        def __init__(self, data, target_name: str, problem_type: str = 'max', manual_input = None,n=5):
            self.df = data
            self.target_name = target_name
            self.problem_type = problem_type
            self.manual_input = manual_input
            self.n = n

        def get_bound(self) -> [float, float]:
            '''
            Function to automatically get the bounds of the 
            target tag based on problem type
            ----------
            df : pd.DataFrame
                    data of intrest
            target_name: str
                    column of interest
            problem_type:str
                what kind of inequality do you want
                "max": >= 
                "min":<= 
                "range":>= and <=
            manual_input:[float,float]
                manual threshold to compare to
            Returns
            -------
            lower : float
                lower bound of the target tag
            upper : float
                upper bound of the target tag
            '''
            print('Finding bounds...')

            # determining Q1,Q3 and iqr
            quartiles = self.df[self.target_name].quantile([0.25, 0.75])
            iqr = quartiles[0.75] - quartiles[0.25]

            if self.problem_type == 'max':
                lower = quartiles[0.25] - (1.5*iqr)
                upper = max(self.df[self.target_name])   
            elif self.problem_type == 'min':
                lower = min(self.df[self.target_name])
                upper = quartiles[0.75] + (1.5*iqr)
                # making sure there are no negative values
                if lower < 0:
                    lower = 0 
            elif self.problem_type == 'range':
                lower = quartiles[0.25] - (1.5*iqr)
                # making sure there are no negative values
                if lower < 0:
                    lower = 0 
                upper = quartiles[0.75] + (1.5*iqr)
            else:
                print('Invalid problem type')

            print('Found Bounds!')
            return lower, upper

        def get_anomalies(self) -> pd.DataFrame:
            '''
            Function to seperate the good and bad outputs from 
            each other in a dataset based on a certain threshold
            ----------
            df : pd.DataFrame
                    data of intrest
            target_name: str
                    column of interest
            problem_type:str
                what kind of inequality do you want
                "max": >= 
                "min":<= 
                "range":>= and <=
            manual_input:[float,float]
                manual threshold to compare to
            Returns
            -------
            optimal : pd.DataFrame
                data of optimal instances
            suboptimal : pd.DataFrame
                data of suboptimal instances
            '''
            # creating the filter column based on given threshold
            if type(self.manual_input) == tuple:
                lower, upper = self.manual_input
            if self.manual_input == None:
                 lower, upper = self.get_bound()
            else: 
                print('invalid data type')
            
            df = self.df.copy()

            if self.problem_type == 'max':
                df['Anomaly'] = np.where(np.greater_equal(self.df[self.target_name],lower), 0, 1)
            elif self.problem_type == 'min':
                df['Anomaly'] = np.where(np.less_equal(self.df[self.target_name],upper), 0, 1)
            elif self.problem_type == 'both':
                df['Anomaly'] = np.where(np.logical_and(np.greater_equal(self.df[self.target_name],lower),np.less_equal(df[self.target_name],upper)), 1, 0)
            else:
                print('ERROR: choose another filter type')

            # splitting the data based on whether or not it is optimal or suboptimal
            optimal = df[df['Anomaly']== 0]
            suboptimal = df[df['Anomaly']== 1]

            return optimal, suboptimal
        
        def anomaly_report(self):
                '''
                Function to automate the creation of 
                an anomaly report for the IES team
                ----------
                df : pd.DataFrame
                    data of intrest
                target_name: str
                    column of interest
                problem_type:str
                    what kind of inequality do you want
                    "max": >= 
                    "min":<= 
                    "range":>= and <=
                manual_input:[float,float]
                    manual threshold to compare to
                Returns
                -------
                optimaloutputtop.csv
                    all time stamps for the topn correlated tags that had optimal outputs
                suboptimaloutputtop10.csv
                    all time stamps for the topn correlated that had suboptimal outputs
                boxplots
                    pngs boxplots of 
                    optimal vs suboptimal outputs for topn tags
                correlations
                    correlation csvs for topn tags 
                '''

                # creating all the folders and subfolders for 
                # the process 
                
                try:
                    os.mkdir('./Data_Cleansing/anomaly_report')
                    os.mkdir('./Data_Cleansing/anomaly_report/xlsx')
                    os.mkdir('./Data_Cleansing/anomaly_report/xlsx/correlations')
                    os.mkdir('./Data_Cleansing/anomaly_report/graphics')
                except OSError as error:
                    print(error)
                
                target = self.df[self.target_name]
                self.df = self.df.drop(self.target_name,axis=1)
                self.df.insert(0,self.target_name,target)
                # feature selecting to find the top 10 most important features
                # (change to the top n/k/j.......whatever)
                args = FeatureSelection(self.df,self.target_name)
                topn = args.correlation_selection().index
                print('Selected Features are' , topn)

                # separtating into optimal and suboptimal outputs 
                # filtering the top 10 most important features 
                # outputing those to .csvs
                p = AnomalyDetection(self.df, self.target_name, self.problem_type, self.manual_input)
                lower, upper = p.get_bound()
                optimaloutput, suboptimaloutput = p.get_anomalies()
                optimaloutputtop = optimaloutput[topn]
                suboptimaloutputtop = suboptimaloutput[topn]

                optimaloutputtop.to_excel('./Data_Cleansing/anomaly_report/xlsx/'+self.target_name+'_toptagsoptimal.xlsx', index=True)
                suboptimaloutputtop.to_excel('./Data_Cleansing/anomaly_report/xlsx/'+self.target_name+'_toptagssuboptimal.xlsx', index=True)

                # making boxplots
                k = len(optimaloutputtop.columns)-1
                if not optimaloutputtop.empty and not suboptimaloutputtop.empty:
                    for i in range(0,k):
                        if i % 2 == 0:
                            optimal = optimaloutputtop.iloc[:,i:i+2]
                            suboptimal = suboptimaloutputtop.iloc[:,i:i+2]
                            vis = BoxPlots(optimal,suboptimal,optimal.columns[0],optimal.columns[1])
                            vis.double_boxplot()
                else:
                    print("Error: The DataFrames optimaloutputtop and/or suboptimaloutputtop are empty.")
                
                # creating correlation csvs
                corr = CorrelationReport(self.df, topn, self.n, self.target_name)
                corr.correlations()


                # Making Basic Stats for IES
                if self.problem_type == 'max':
                     threshtype = 'greater than '
                     bound = str(lower)
                elif self.problem_type == 'min':
                     threshtype = 'less than '
                     bound = str(upper)
                elif self.problem_type == 'both':
                     threshtype = 'between '
                     bound = ''
                else:
                     print('Invalid Threshold type')
                with open('./Data_Cleansing/anomaly_report/stats.txt', 'w') as f:
                    f.write('Date Range: ' + str(min(self.df.index)) +' - '+ str(max(self.df.index)))
                    f.write('\n')
                    f.write('Total number of instances observed by Maestro: ' + str(len(self.df)))
                    f.write('\n')
                    f.write('Total number of operational tags: ' + str(len(self.df.columns)))
                    f.write('\n')
                    f.write('Number of important tags: ' + str(len(topn)))
                    f.write('\n')
                    f.write('Total number of sub-optimal ('+ self.target_name +' '+ threshtype + bound + ') instances: ' + str(len(suboptimaloutput)))
                    f.write('\n')
                    f.write('Total number of optimal (' + self.target_name + ' not '+ threshtype + bound  + ') instances: ' + str(len(optimaloutput)))

                print('Done')
        

            

       

        



