import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Visualization.boxplots import *
from Data_Analyzing.correlation_report import *
from Data_Analyzing.feature_selection import * 
from Data_Cleansing.anomaly_detection import *

class AnomalyReport:
    def __init__(self,data, target_name: str, problem='min', manual_input=None, n=5):
        self.df = data
        self.target_name = target_name
        self.problem = problem
        self.manual_input = manual_input
        self.n = n
    
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
                    os.mkdir('./Data_Cleansing/anomaly_report')
                    os.mkdir('./Data_Cleansing/anomaly_report/xlsx')
                    os.mkdir('./Data_Cleansing/anomaly_report/graphics')
                except OSError as error:
                    print(error)
                

                #feature selecting to find the top 10 most important features
                #(change to the top n/k/j.......whatever)
                args = FeatureSelection(self.df,self.target_name)
                topn = args.correlation_selection().index
                print('Selected Features are' , topn)

                #separtating into good and bad outputs 
                #filtering the top 10 most important features 
                #outputing those to .csvs
                p = AnomalyDetection(self.df, self.target_name, self.problem, self.manual_input)
                lower, upper = p.get_bound()
                goodoutput, badoutput = p.get_anomalies()
                goodoutputtop = goodoutput[topn]
                badoutputtop = badoutput[topn]

                goodoutputtop.to_excel('./Data_Cleansing/anomaly_report/xlsx/'+self.target_name+'_toptagsgood.xlsx', index=False)
                badoutputtop.to_excel('./Data_Cleansing/anomaly_report/xlsx/'+self.target_name+'_toptagsbad.xlsx', index=False)

                #making boxplots
                k = len(goodoutputtop.columns)-1
                if not goodoutputtop.empty and not badoutputtop.empty:
                    for i in range(0,k):
                        if i % 2 == 0:
                            good = goodoutputtop.iloc[:,i:i+2]
                            bad = badoutputtop.iloc[:,i:i+2]
                            vis = BoxPlots(good,bad,good.columns[0],good.columns[1])
                            vis.double_boxplot()
                #Rest of the code for creating boxplots
                else:
                    print("Error: The DataFrames goodoutputtop and/or badoutputtop are empty.")
                
                #creating correlation csvs
                corr = CorrelationReport(self.df, topn, self.n, self.target_name)
                corr.correlations()

                if self.problem == 'max':
                     threshtype = 'greater than '
                     bound = str(lower)
                elif self.problem == 'min':
                     threshtype = 'less than '
                     bound = str(upper)
                elif self.problem == 'both':
                     threshtype = 'between '
                     bound = ''
                else:
                     print('Invalid Threshold type')
                with open('./Data_Cleansing/anomaly_report/stats.txt', 'w') as f:
                    f.write('Date Range: ' + str(min(self.df.index)) +' - '+ str(max(self.df.index)))
                    f.write('\n')
                    f.write('Total number of instances observed by Maestro (Date Range): ' + str(len(self.df)))
                    f.write('\n')
                    f.write('Total number of operational tags: ' + str(len(self.df.columns)))
                    f.write('\n')
                    f.write('Total number of sub-optimal ('+ self.target_name +' '+ threshtype + bound + ') instances: ' + str(len(badoutput)))
                    f.write('\n')
                    f.write('Total number of optimal (' + bound + ' not '+ threshtype + self.target_name + ') instances: ' + str(len(goodoutput)))

                print('Done')