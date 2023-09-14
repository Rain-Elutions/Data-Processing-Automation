import pandas as pd 
import plotly.express as px
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import correlation

class CorrelationTypes:
    def __init__(self, data, topn, target_name):
         self.df = data
         self.topn = topn
         self.target_name = target_name
    
    def get_redundant_pairs(self):
         df = self.df.select_dtypes(include=np.number)
         pairs_to_drop =set()
         cols = df.columns
         for i in range(0,df.shape[1]):
            for j in range(0,i+1):
                pairs_to_drop.add((cols[i],cols[j]))
         return pairs_to_drop
    
    def get_correlations(self):
         df = self.df.select_dtypes(include=np.number)
         au_corr = df.corr().dropna(how='all').dropna(axis=1,how='all')
         return au_corr
    
    def non_linear(self):
         
         def calculate_MI(df: pd.DataFrame) -> pd.Series:
            mi_values = []
            for col1 in df.columns:
                mi_row = []
                for col2 in df.columns:
                    v1 = np.array(df[col1])
                    v2 = np.array(df[col2])
                    mi = mutual_info_regression(v1.reshape(-1, 1), v2)[0]
                    mi_row.append(mi)
                
                mi_values.append(mi_row)
            
            mi_matrix = pd.DataFrame(mi_values, columns=df.columns, index=df.columns).dropna(how='all').dropna(axis=1,how='all')
            return mi_matrix
         df = self.df

         # Calculate the mean for numeric columns
         numeric_columns = df.select_dtypes(include=[np.number]).columns
         spearmancorr = df[numeric_columns].corr(method='spearman').dropna(how='all').dropna(axis=1,how='all')
         # Calculate mutual information matrix
         df = df[numeric_columns].dropna(axis=1,how='all')
         mi_matrix = calculate_MI(df)

         # Export the mutual information matrix to a CSV file
         mi_matrix.to_csv('./mutual_information_matrix.csv', index=True)
         return mi_matrix,spearmancorr

    def top_correlations(self):
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