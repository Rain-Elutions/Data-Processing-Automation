import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from Data_Visualization.plot_types import BoxPlots
from Data_Analyzing.correlation_analysis import CorrelationTypes
from Data_Analyzing.feature_selection import FeatureSelection


def create_directory_with_numbered_suffix(base_path, directory_name):
    new_directory_name = f'{directory_name}'
    new_directory_path = os.path.join(base_path, new_directory_name)
    try:
        os.mkdir(new_directory_path)
        return new_directory_name, new_directory_path
    except OSError:
        print(f'{new_directory_path} already exists, trying again...')
        return new_directory_name, new_directory_path


class AnomalyDetection:
        """
        Class to automate the detection of anomalies 
        across multiple datasets, and create an anomaly report
        for the IES team
        ----------
        data : pd.DataFrame
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
        manual_thresh:float 
            0<x<1 threshold for determining anomalous 
            instances 
        """
        def __init__(self, data, target_name: str = '', problem_type: str = 'max', manual_input=None, manual_thresh=None):
            self.df = data
            self.target_name = target_name
            self.problem_type = problem_type
            self.manual_input = manual_input
            self.manual_thresh = manual_thresh

        def __get_bound(self):
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
            if self.manual_thresh == None:
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
            else:
                custom_quartile = self.df[self.target_name].quantile(self.manual_thresh)
                if self.problem_type == 'max':
                    lower = custom_quartile
                    upper = max(self.df[self.target_name])   
                elif self.problem_type == 'min':
                    lower = min(self.df[self.target_name])
                    upper = custom_quartile
                    # making sure there are no negative values
                    if lower < 0:
                        lower = 0 
                else:
                    print('Invalid problem type')

            print('Found Bounds!')

            return lower, upper

        def __get_anomalies(self):
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
                 lower, upper = self.__get_bound()
            else: 
                print('invalid data type')
            
            df = self.df.copy()

            # finding the actual anomalies 
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

            return optimal, suboptimal, lower , upper
        
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
                # the report 

                base_directory = './Data_Cleansing/'

                # if the target name has a : in it, replace it with _ when creating the directory
                target_name = self.target_name
                if ":" in self.target_name:
                    target_name = self.target_name.replace(":", "_")

                new_directory_name, new_directory_path = create_directory_with_numbered_suffix(base_directory, target_name + '_anomaly_report')
                try:
                    os.mkdir(os.path.join(new_directory_path, 'xlsx'))
                    os.mkdir(os.path.join(new_directory_path, 'xlsx', 'correlations'))
                    os.mkdir(os.path.join(new_directory_path, 'graphics'))
                except OSError as error:
                    print(error)

                #splitting the data 
                target = self.df[self.target_name]
                self.df = self.df.drop(self.target_name,axis=1)
                self.df.insert(0,self.target_name,target)


                # feature selecting to find the top n most important features
                args = FeatureSelection(self.df,self.target_name)
                topn , selected_features = args.correlation_selection()
                topn_list = list(topn)
                print('Selected Features are' , ', '.join(topn_list))

                # separtating into optimal and suboptimal outputs 
                # filtering the top n most important features 
                # outputing those to .csvs
                optimaloutput, suboptimaloutput, lower, upper = self.__get_anomalies()
                optimaloutputtop = optimaloutput[topn]
                suboptimaloutputtop = suboptimaloutput[topn]

                optimaloutputtop.to_excel('./Data_Cleansing/'+ new_directory_name + '/xlsx/' + target_name + '_toptagsoptimal.xlsx', index=True)
                suboptimaloutputtop.to_excel('./Data_Cleansing/'+ new_directory_name + '/xlsx/' + target_name + '_toptagssuboptimal.xlsx', index=True)

                # making boxplots
                k = len(optimaloutputtop.columns)-1
                if not optimaloutputtop.empty and not suboptimaloutputtop.empty:
                    for i in range(0,k):
                        if i % 2 == 0:
                            optimal = optimaloutputtop.iloc[:,i:i+2]
                            suboptimal = suboptimaloutputtop.iloc[:,i:i+2]
                            vis = BoxPlots(optimal, suboptimal, target_name, optimal.columns[0], optimal.columns[1])
                            vis.double_boxplot()
                else:
                    print("Error: The DataFrames optimaloutputtop and/or suboptimaloutputtop are empty.")
                
                #creating correlation csvs
                corr = CorrelationTypes(self.df, topn, self.target_name)
                corr.top_correlations()


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
                with open('./Data_Cleansing/'+ new_directory_name +'/stats.txt', 'w') as f:
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
        

            

       

        



