import pandas as pd 
import plotly.express as px

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
            toprelationspearson = allrealtionspearson[self.topn]
            tagpearson = allrealtionspearson[self.target_name].dropna().sort_values(ascending=False)

            #Making correlation plot for top and bottom n linear correlations
            self.target_nametop = list(tagpearson[:self.n].index)
            self.target_namebottom = list(tagpearson[(len(tagpearson)-self.n):len(tagpearson)].index)
            corr1 = allrealtionspearson[self.target_nametop+self.target_namebottom].filter(self.target_nametop+self.target_namebottom,axis=0)
            fig = px.imshow(corr1, color_continuous_scale='Blues')
            fig.write_html("./Data_Cleansing/anomaly_report/graphics/" + self.target_name + ".html")
            

            #merging the top corelations together
            k = len(toprelationspearson.columns)
            for i in range(0,k):
                pearson = toprelationspearson.iloc[:,i:i+1].sort_values(by=toprelationspearson.columns[i],ascending=False)
                pearson.to_excel('./Data_Cleansing/anomaly_report/xlsx/correlations/' + pearson.columns[0] +' corr.xlsx')