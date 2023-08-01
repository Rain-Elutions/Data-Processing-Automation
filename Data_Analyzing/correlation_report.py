import pandas as pd 
import plotly_express as px
class CorrelationReport:
    def __init__(self,data,topn,n,target_name):
         self.df = data
         self.topn = topn
         self.n = n
         self.target_name = target_name
    
    def correlations(self):
            '''
            Function to create correlation csvs
            for the top 10 important features
            ----------
            df : pd.DataFrame
                data of intrest
            top10: list 
                top 10 most important features
            taget: self.target_name of intrest
            n: int 
                number of top and bottom linear 
                correlation to filter
            Returns
            ----------
            pearson csvs for top10 tags and self.target_name
            '''
            #Making pearson correlations
            allrealtionspearson = self.df.corr()
            top10relationspearson = allrealtionspearson[self.topn]
            tagpearson = allrealtionspearson[self.target_name].dropna().sort_values(ascending=False)
            tagpearson.to_excel('./anomaly_report/xlsx/' + self.target_name +' corr.xlsx',index = False)

            #Making correlation plot for top and bottom 5 linear correlations
            self.target_nametop = list(tagpearson[:self.n].index)
            self.target_namebottom = list(tagpearson[(len(tagpearson)-self.n):len(tagpearson)].index)
            corr1 = allrealtionspearson[self.target_nametop+self.target_namebottom].filter(self.target_nametop+self.target_namebottom,axis=0)
            fig = px.imshow(corr1, color_continuous_scale='Blues')
            fig.write_html("./anomaly_report/graphics/" + self.target_name + ".html")
            

            #merging the top corelations together
            k = len(top10relationspearson.columns)-1
            for i in range(0,k):
                pearson = top10relationspearson.iloc[:,i:i+1].sort_values(by=top10relationspearson.columns[i],ascending=False)
                pearson.to_excel('./anomaly_report/xlsx/' + pearson.columns[0] +' corr.xlsx')