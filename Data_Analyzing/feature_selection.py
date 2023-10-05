import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from xgboost import XGBRegressor
from BorutaShap import BorutaShap
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Visualization.eda import EDA_Visualization


class FeatureSelection:
    '''
    Feature selection class

    Parameters:
    - data: the input DataFrame
    - target_name: the name of the target column
    '''
    def __init__(self, data,target_name):
        self.data = data
        self.target_name = target_name

    def correlation_selection(self, threshold: float=.5) -> list[str]:
        '''
        feature selection based on correlation to target

        Parameters:
        - threshold: the threshold of correlation to be selected

        Returns:
        - selected_feats: the columns with correlation to target greater than threshold
        - data: the data with selected features
        '''

        # threshold being .5 comes from IES
        corr_matrix = self.data.corr()
        # get columns with correlation to target greater than threshold
        target_corr = corr_matrix[self.target_name]
        target_corr_positive = target_corr[target_corr > threshold]
        target_corr_negative = target_corr[target_corr < (-1)*threshold]
        selected_feats = pd.concat([target_corr_positive, target_corr_negative])
        selected_feats = selected_feats.to_frame()
        return selected_feats, self.data[selected_feats.columns[0]]

    def dummy_feature_importance(self, select_num: int = None, iter: int = 20) -> tuple[list[str], pd.DataFrame]:
        '''
        Feature selection based on dummy feature importance, get from XGBoost

        Parameters:
        - select_num: the number of features to be selected
        - iter: the number of iterations to get the average feature importance

        Returns:
        - selected_feats: the selected features
        - data: the data with selected features
        '''
        feat_importance_dict = {col: 0 for col in self.data.columns}
        feat_importance_dict['dummy'] = 0
        feat_importance_dict.pop(self.target_name)

        # iterate to get the average feature importance using XBRegressor
        y = self.data[self.target_name]
        for i in tqdm(range(iter)):
            X = self.data.drop(self.target_name,axis=1)
            np.random.seed(i)
            X['dummy'] = np.random.rand(len(X))

            xgb = XGBRegressor(random_state=123)
            xgb.fit(X, y)
            
            scaler = MinMaxScaler(feature_range=(0, 10000000))
            scaled_importances = scaler.fit_transform(xgb.feature_importances_.reshape(-1, 1)).reshape(-1)

            for feat in feat_importance_dict.keys():
                feat_importance_dict[feat] += scaled_importances[X.columns.get_indexer([feat])[0]]

        for feat in feat_importance_dict.keys():
            feat_importance_dict[feat] /= iter
        
        ev = EDA_Visualization(self.data)
        ev.visualize_feature_importance(feat_importance_dict, ['dummy'])

        # select the highest select_num features
        if select_num:
            # remove the dummy feature
            feat_importance_dict.pop('dummy')
            selected_feats = sorted(feat_importance_dict.items(), key=lambda x: x[1], reverse=True)[:select_num]
            selected_feats = [feat[0] for feat in selected_feats]
        else:
            # select features more important than dummy feature
            selected_feats = [feat for feat in feat_importance_dict.keys() if feat_importance_dict[feat] > feat_importance_dict['dummy']]

        return selected_feats, self.data[selected_feats]
    
    def borutashap_feature_selection(self) -> tuple[list[str], pd.DataFrame]:
        '''
        Feature selection based on Boruta-Shap

        Returns:
        - selected_feats: the selected features
        - data: the data with selected features
        '''

        # Initialize Boruta-Shap feature selection method
        feature_selection_model = XGBRegressor(random_state=123) 
        Feature_Selector = BorutaShap(model=feature_selection_model, 
            importance_measure='shap', 
            classification=False)
        X = self.data.drop(self.target_name,axis=1)

        # Fit the Boruta-Shap feature selection model, and get all relevant features
        Feature_Selector.fit(X=X, y=self.data[self.target_name], n_trials=100, sample=False, 
                            train_or_test = 'train', normalize=False, 
                            verbose=True, random_state=123)

        # Get the selected features
        selected_feats = list()
        selected_feats.append(sorted(Feature_Selector.Subset().columns))
        print(f"Selected features are: {selected_feats[0]}")

        return selected_feats[0], self.data[selected_feats[0]]