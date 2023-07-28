import os
from visualization import *
from feature_selection import * 

class correlation_analysis:
        def __init__(self,data,targ: str,fname='./', problem = 'max', lower = 0, upper= 0, n = 5,manual = None):
            self.df = data
            self.targ = targ
            self.fname = fname
            self.problem = problem
            self.lower = lower
            self.upper = lower
            self.n = n
            self.manual = manual

        def bounds(self):

            quartiles = self.df[self.targ].quantile([0.25, 0.75])
            iqr = quartiles[0.75] - quartiles[0.25]
            if self.problem == 'max':
                lower = min(self.df[self.targ])
                upper = quartiles[0.75] + (1.5*iqr)
            if self.problem == 'min':
                lower = quartiles[0.25] - (1.5*iqr)
                if lower < 0:
                    lower = 0 
                upper = max(self.df[self.targ])
            if self.problem == 'range':
                lower = quartiles[0.25] - (1.5*iqr)
                if lower < 0:
                    lower = 0 
                upper = quartiles[0.75] + (1.5*iqr)
            else:
                print('Invalid problem type')

            return lower,upper

        def good_bad_data(self)->pd.DataFrame:
            '''
            Function to seperate the good and bad 
            outputs from each other in a dataset
            based on a certain threshold
            ----------
            df : pd.DataFrame
                data of intrest
            self.targ: str
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
            df = self.df.copy()

            if self.problem == 'max':
                df['Status'] = np.where(np.greater_equal(self.df[self.targ],self.lower), 1, 0)
            elif self.problem  == 'min':
                df['Status'] = np.where(np.less_equal(self.df[self.targ],self.upper), 1, 0)
            elif self.problem  == 'both':
                df['Status'] = np.where(np.logical_and(np.greater_equal(self.df[self.targ],self.lower),np.less_equal(df[self.targ],self.upper)), 1, 0)
            else:
                print('ERROR: choose another filter type')

            #splitting the data based on whether or not it is good or bad
            good= df[df['Status']== 1]
            bad = df[df['Status']== 0]

            return good,bad


        def anomaly_report(self):
            '''
            Function to automate the creation of 
            an anomaly report for the IES team
            ----------
            df : pd.DataFrame
                data of intrest
            targ: str
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
            n: int 
                number of top and bottom linear 
                correlation to filter
            autothresh: bool
                whether or not to have python find the threshold
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
            
            #auto selecting the threshold of the self.targ variable
           
            if type(self.manual) == tuple:
                lower,upper = self.manual
            if self.manual == None:
                 lower,upper = self.bounds()
            else: 
                print('invalid data type')

            #feature selecting to find the top 10 most important features
            #(change to the top n/k/j.......whatever)
            args = feature_selector(self.df,self.targ,file_name=self.fname)
            top10 = args.boruta()
            

            #separtating into good and bad outputs 
            #filtering the top 10 most important features 
            #outputing those to .csvs
            print(top10)
            goodoutput,badoutput = self.good_bad_data()
            goodoutputtop10 = goodoutput[top10]
            badoutputtop10 = badoutput[top10]

            goodoutputtop10.to_csv('./anomaly_report/csvs/'+self.targ+'top10good.csv')
            badoutputtop10.to_csv('./anomaly_report/csvs/'+self.targ+'top10bad.csv')

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
            print('Total number of sub-optimal ('+ self.targ +' '+ self.threshtype + ' than '+ str(self.thresh1) + ') instances: ' + str(len(badoutput)))
            print('Total number of optimal (' + str(self.thresh1) + ' not '+ self.threshtype + ' than '+ self.targ + ') instances: ' + str(len(goodoutput)))

        def correlations(self,top10):
            '''
            Function to create correlation csvs
            for the top 10 important features
            ----------
            df : pd.DataFrame
                data of intrest
            top10: list 
                top 10 most important features
            taget: self.targ of intrest
            n: int 
                number of top and bottom linear 
                correlation to filter
            Returns
            ----------
            pearson csvs for top10 tags and self.targ
            '''
            #Making pearson correlations
            allrealtionspearson = self.df.corr()
            top10relationspearson = allrealtionspearson[top10]
            tagpearson = allrealtionspearson[self.targ].dropna().sort_values(ascending=False)
            tagpearson.to_csv('./anomaly_report/csvs/' + self.targ +' corr.csv')

            #Making correlation plot for top and bottom 5 linear correlations
            self.targtop = list(tagpearson[:self.n].index)
            self.targbottom = list(tagpearson[(len(tagpearson)-self.n):len(tagpearson)].index)
            corr1 = allrealtionspearson[self.targtop+self.targbottom].filter(self.targtop+self.targbottom,axis=0)
            fig = px.imshow(corr1, color_continuous_scale='Blues')
            fig.write_html("./anomaly_report/graphics/" + self.targ + ".html")
            

            #merging the top 10 corelations together
            for i in range(0,10):
                pearson = top10relationspearson.iloc[:,i:i+1].sort_values(by=top10relationspearson.columns[i],ascending=False)
                pearson.to_csv('./anomaly_report/csvs/' + pearson.columns[0] +' corr.csv')

            

       

        



