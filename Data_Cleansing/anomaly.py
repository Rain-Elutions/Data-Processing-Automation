import os
from boxplots import *
from feature_selection import * 

class AnomalyDetection:
        def __init__(self,data,target_name: str, problem = 'max', lower = 0, upper= 0,manual_input = None):
            self.df = data
            self.target_name = target_name
            self.problem = problem
            self.lower = lower
            self.upper = lower
            self.manual_input = manual_input

        def bounds(self):

            quartiles = self.df[self.target_name].quantile([0.25, 0.75])
            iqr = quartiles[0.75] - quartiles[0.25]
            if self.problem == 'max':
                lower = min(self.df[self.target_name])
                upper = quartiles[0.75] + (1.5*iqr)
            if self.problem == 'min':
                lower = quartiles[0.25] - (1.5*iqr)
                if lower < 0:
                    lower = 0 
                upper = max(self.df[self.target_name])
            if self.problem == 'range':
                lower = quartiles[0.25] - (1.5*iqr)
                if lower < 0:
                    lower = 0 
                upper = quartiles[0.75] + (1.5*iqr)
            else:
                print('Invalid problem type')

            return lower,upper

        def detect_anomaly(self)->pd.DataFrame:
            '''
            Function to seperate the good and bad 
            outputs from each other in a dataset
            based on a certain threshold
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
            #creating the filter column based on given threshold
            if type(self.manual_input) == tuple:
                lower,upper = self.manual_input
            if self.manual_input== None:
                 lower,upper = self.bounds()
            else: 
                print('invalid data type')
            
            df = self.df.copy()

            if self.problem == 'max':
                df['Anomaly'] = np.where(np.greater_equal(self.df[self.target_name],self.lower), 1, 0)
            elif self.problem  == 'min':
                df['Anomaly'] = np.where(np.less_equal(self.df[self.target_name],self.upper), 1, 0)
            elif self.problem  == 'both':
                df['Anomaly'] = np.where(np.logical_and(np.greater_equal(self.df[self.target_name],self.lower),np.less_equal(df[self.target_name],self.upper)), 1, 0)
            else:
                print('ERROR: choose another filter type')

            #splitting the data based on whether or not it is good or bad
            good= df[df['Anomaly']== 1]
            bad = df[df['Anomaly']== 0]

            return good,bad


        def anomaly_report(self):
            '''
            Function to automate the creation of 
            an anomaly report for the IES team
            ----------
            df : pd.DataFrame
                data of intrest
            target_name: str
                column of interest
            self.fname: str
                file name to save boruta files to
            threshtype:str
                what kind of inequality do you want
                "greater": >= 
                "lesser":<= 
                "both":>= and <=
            thresh1:int
                threshold to compare to. 
                When threshtype = "both" >=
            thresh2:int
                threshold to compare to. 
                When threshtype = "both" <=
            Returns
            -------
            top10: list
                list of the top 10
                most important features
            goodoutputtop10.csv
                all time stamps for the top10 tags that had good outputs
            badoutputtop10.csv
                all time stamps for the top10 tags that had bad outputs
            boxplots
                pdfs and pngs of boxplots of 
                good vs bad outputs for top10 tags
            correlations
                pearson and spearman csvs for top10 tags 
            '''
            #creating all the folders and subfolders for 
            #the process 
            
            try:
                os.mkdir('./anomaly_report')
                os.mkdir('./anomaly_report/csvs')
                os.mkdir('./anomaly_report/graphics')
            except OSError as error:
                print(error)
            

            #feature selecting to find the top 10 most important features
            #(change to the top n/k/j.......whatever)
            args = feature_selector(self.df,self.target_name,file_name=self.fname)
            top10 = args.boruta()
            

            #separtating into good and bad outputs 
            #filtering the top 10 most important features 
            #outputing those to .csvs
            print(top10)
            goodoutput,badoutput = self.detect_anomaly()
            goodoutputtop10 = goodoutput[top10]
            badoutputtop10 = badoutput[top10]

            goodoutputtop10.to_excel('./anomaly_report/xlsx/'+self.target_name+'toptagsgood.xlsx',index=False)
            badoutputtop10.to_excel('./anomaly_report/xlsx/'+self.target_name+'toptagsbad.xlsx',index=False)

            #making boxplots
            for i in range(0,10):
                if i % 2 == 0:
                    good = goodoutputtop10.iloc[:,i:i+2]
                    bad = badoutputtop10.iloc[:,i:i+2]
                    vis = visualizations(good,bad,good.columns[0],good.columns[1])
                    vis.double_boxplot()

            
            #creating correlation csvs
            self.correlations(top10)

            print('Total number of instances observed by Maestro (Date Range): ' + str(len(self.df)) + str(min(self.df.index)) +' - '+ str(max(self.df.index)))
            print('Total number of operational tags: ' + str(len(self.df.columns)))
            print('Total number of sub-optimal ('+ self.target_name +' '+ self.threshtype + ' than '+ str(self.thresh1) + ') instances: ' + str(len(badoutput)))
            print('Total number of optimal (' + str(self.thresh1) + ' not '+ self.threshtype + ' than '+ self.target_name + ') instances: ' + str(len(goodoutput)))

        

            

       

        



