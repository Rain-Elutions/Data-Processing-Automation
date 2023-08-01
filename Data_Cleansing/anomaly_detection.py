import pandas as pd 
import numpy as np

class AnomalyDetection:
        def __init__(self, data, target_name: str, problem_type: str = 'min', manual_input = None):
            self.df = data
            self.target_name = target_name
            self.problem_type = problem_type
            self.manual_input = manual_input

        def get_bound(self) -> [float, float]:
            print('Finding bounds...')
            quartiles = self.df[self.target_name].quantile([0.25, 0.75])
            iqr = quartiles[0.75] - quartiles[0.25]

            if self.problem_type == 'max':
                lower = quartiles[0.25] - (1.5*iqr)
                upper = max(self.df[self.target_name])   
            elif self.problem_type == 'min':
                lower = min(self.df[self.target_name])
                upper = quartiles[0.75] + (1.5*iqr)
                if lower < 0:
                    lower = 0 
            elif self.problem_type == 'range':
                lower = quartiles[0.25] - (1.5*iqr)
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
            self.target_name: str
                column of interest
            goodthreshold: int
                threshold on which to filter
            self.threshtype:str
                what kind of inequality do you want
                "greater": >= 
                "lesser":<= 
                "both":>= and <=
            self.thresh1:int
                threshold to compare to. 
                When threshtype = "both" >=
            self.thresh2:int
                threshold to compare to. 
                When threshtype = "both" <=
            Returns
            -------
            good : pd.DataFrame
                data of good bathces 
            bad : pd.DataFrame
                data of bad bathces 
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
                df['Anomaly'] = np.where(np.greater_equal(self.df[self.target_name],lower), 1, 0)
            elif self.problem_type == 'min':
                df['Anomaly'] = np.where(np.less_equal(self.df[self.target_name],upper), 1, 0)
            elif self.problem_type == 'both':
                df['Anomaly'] = np.where(np.logical_and(np.greater_equal(self.df[self.target_name],lower),np.less_equal(df[self.target_name],upper)), 1, 0)
            else:
                print('ERROR: choose another filter type')

            # splitting the data based on whether or not it is good or bad
            good = df[df['Anomaly']== 1]
            bad = df[df['Anomaly']== 0]

            return good, bad

        

            

       

        



