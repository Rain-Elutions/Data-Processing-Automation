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
    
    def get_correlations(self):
         '''
        Calculate the correlations of each column for linear relations

        Parameters:
        -data: the input Dataframe
        '''
         #selecting only numeric columns 
         df = self.df.select_dtypes(include=np.number)

         #creating the correlations
         fullcorr = df.corr()
         augcorr = fullcorr.dropna(how='all').dropna(axis=1,how='all') # drop rows/columns where all correlations are NA
         return augcorr
    
    # def calculate_MI(self,df: pd.DataFrame) -> pd.Series:
    #         '''
    #         Calculate the mutual information of a df in order to determine nonlinear relations

    #         Parameters:
    #         -data: the input Dataframe
    #         '''
    #         mi_values = []
    #         for col1 in df.columns:
    #             mi_row = []
    #             for col2 in df.columns:
    #                 v1 = np.array(df[col1])
    #                 v2 = np.array(df[col2])
    #                 mi = mutual_info_regression(v1.reshape(-1, 1), v2)[0]
    #                 mi_row.append(mi)
                
    #             mi_values.append(mi_row)
    #         mi_matrixfull =  pd.DataFrame(mi_values, columns=df.columns, index=df.columns)
    #         mi_matrixaug = mi_matrixfull.dropna(how='all').dropna(axis=1,how='all')# drop rows/columns where all correlations are NA
     
    #         return mi_matrixaug
    
    def non_linear(self):
         '''
        Calculate the correlations of each column for nonlinear relations

        Parameters:
        -data: the input Dataframe
        '''
         # Calculate the mean for numeric columns
         numeric_columns =self.df.select_dtypes(include=[np.number]).columns
         spearmancorr = self.df[numeric_columns].corr(method='spearman').dropna(how='all').dropna(axis=1,how='all')

         # Calculate mutual information matrix
         df = self.df[numeric_columns].dropna(axis=1,how='all')
         if df.isnull().values.any() == False:
            mi_matrixaug = self.calculate_MI(df)
         else:
              mi_matrixaug = None
         return  mi_matrixaug,spearmancorr

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
            toprelations = allrealtions[self.topn.columns[0]]
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
            toprelations = toprelations.to_frame()
            k = len(toprelations.columns)
            print(toprelations.columns)
            for i in range(0,k):
                tagcolumncorr = toprelations.iloc[:,i:i+1].sort_values(by=toprelations.columns[i],ascending=False)
                top_col = tagcolumncorr.columns[0]
                print(top_col)
                if ":" in top_col:
                    top_col = top_col.replace(":", "_")
                tagcolumncorr.to_excel('./Data_Cleansing/'+ self.target_name +'_anomaly_report/xlsx/correlations/' + top_col +' corr.xlsx')