import pandas as pd 
import plotly.express as px
import numpy as np
from minepy import MINE
from sklearn.feature_selection import mutual_info_regression
from scipy.spatial.distance import correlation

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
    
    def non_linear(self):
         def calculate_dist_corr(df: pd.DataFrame) -> float:
            dist_corr_matrix = correlation(df.values.T)
            dist_corr_df = pd.DataFrame(dist_corr_matrix, columns=df.columns, index=df.columns)
            return dist_corr_df
         
         dist_corr_matrix = calculate_dist_corr(self.df)

         # Export the findings to a CSV file
         dist_corr_matrix.to_csv('./distance_correlation_matrix.csv', index=True)

         print("Distance correlation matrix:")
         print(dist_corr_matrix)
         

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
            
            mi_matrix = pd.DataFrame(mi_values, columns=df.columns, index=df.columns)
            return mi_matrix
         df = pd.DataFrame(self.df)

         # Calculate mutual information matrix
         mi_matrix = calculate_MI(df)

         # Export the mutual information matrix to a CSV file
         mi_matrix.to_csv('./mutual_information_matrix.csv', index=True)

         print("Mutual Information Matrix:")
         print(mi_matrix)

         def compute_MIC(v1, v2):
                mine = MINE(alpha=0.6, c=15, est='mic_approx')
                mine.compute_score(v1, v2)
                return round(mine.mic(), 2)

         def calculate_MIC_matrix(df):
             columns = df.columns
             mic_matrix = pd.DataFrame(index=columns, columns=columns)

             for col1 in columns:
                 for col2 in columns:
                     mic_value = compute_MIC(df[col1], df[col2])
                     mic_matrix.loc[col1, col2] = mic_value

             return mic_matrix
         mic_matrix = calculate_MIC_matrix(df)

         # Export the MIC matrix to a CSV file
         mic_matrix.to_csv('./mic_matrix.csv')

         print("MIC Matrix:")
         print(mic_matrix)

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