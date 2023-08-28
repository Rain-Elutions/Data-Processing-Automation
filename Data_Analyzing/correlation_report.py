import pandas as pd 
import plotly.express as px

class CorrelationReport:
    def __init__(self, data, topn, target_name):
         self.df = data
         self.topn = topn
         self.target_name = target_name
    
    def get_redundant_pairs(self):
         pairs_to_drop =set()
         cols = self.df.columns
         for i in range(0,self.df.shape[1]):
            for j in range(0,i+1):
                pairs_to_drop.add((cols[i],cols[j]))
         return pairs_to_drop
    
    def get_correlations(self):
         au_corr = self.df.corr().unstack()
         labels_to_drop = self.get_redundant_pairs(self.df)
         au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
         return au_corr
              
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
            csvs for top10 tags and self.target_name
            '''
            # Making correlations
            # filtering the top relations
            # sorting the correlations
            allrealtions = self.df.corr()
            toprelations = allrealtions[self.topn]
            tag = allrealtions[self.target_name].dropna().sort_values(ascending=False)

            #Making correlation matrix for top and bottom n linear correlations
            self.target_nametop = list(tag[:5].index)
            self.target_namebottom = list(tag[(len(tag)-5):len(tag)].index)
            corr1 = allrealtions[self.target_nametop+self.target_namebottom].filter(self.target_nametop+self.target_namebottom,axis=0)
            fig = px.imshow(corr1, color_continuous_scale='Blues')
            
            if ":" in self.target_name:
                self.target_name = self.target_name.replace(":","_")
            fig.write_html('./Data_Cleansing/'+ self.target_name +"_anomaly_report/graphics/" + self.target_name + ".html")
            
            #merging the top corelations together
            k = len(toprelations.columns)
            print(toprelations.columns)
            for i in range(0,k):
                tagcolumncorr = toprelations.iloc[:,i:i+1].sort_values(by=toprelations.columns[i],ascending=False)
                top_col = tagcolumncorr.columns[0]
                print(top_col)
                if ":" in top_col:
                    top_col = top_col.replace(":", "_")
                tagcolumncorr.to_excel('./Data_Cleansing/'+ self.target_name +'_anomaly_report/xlsx/correlations/' + top_col +' corr.xlsx')