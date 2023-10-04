import pandas as pd
from box import Box
with open('config.yaml') as f:
    cfg = Box.from_yaml(f.read())
import warnings
warnings.filterwarnings('ignore')
from Data_Exploration.data_exploration_main import DataExploration
from Data_Cleansing.data_cleansing_main import DataCleansing
from Data_Preprocessing.data_preprocessing_main import DataPreprocessing
from Data_Preprocessing.feature_engineering import FeatureEngineering
from Data_Analyzing.data_analysis_main import DataAnalysis
from Data_Analyzing.feature_selection import FeatureSelection
from Data_Visualization.eda import EDA_Visualization

class DataProcessing:
    def __init__(self, data_source: str=None, target: str = None,target_list: list = None,problem_type: str = None):
        self.data_source = data_source
        self.target = target
        self.target_list = target_list
        self.problem_type = problem_type

    def pipeline(self, optional: bool=None, resample: bool=None,timescale: str = 'h', engineering: str = None, feature_selector: str = 'boruta'):
        data_exp = DataExploration()
        df = data_exp.load_data(self.data_source, parse_dates=False, index_col=0)
        data_exp.get_data_size()
        data_exp.get_data_type(df,df.columns[0])
        data_exp.summarize_data_type()
        data_exp.summarize_missing_data()

        eda_vis = EDA_Visualization(df)
        eda_vis.visualize_missing_data()

        # DataCleansing Module
        data_cleansing = DataCleansing(df)

        df = data_cleansing.remove_duplicates(df)
        data_exp.summarize_missing_data(df)

        df = data_cleansing.handle_missing_values(df, self.target_list, drop_thresh=0.5, fill_missing_method='mean')
        data_exp.summarize_missing_data(df)

        if self.optional == True:
            data_cleansing.generate_anomaly_report(df, self.target, self.problem_type)
            data_cleansing.detect_outliers(df, col_name=df.columns[10], threshold=3)

        # Encoding & Scaling
        dp = DataPreprocessing(df, [self.target_list])
        df = dp.feature_encoding()
        
        if optional == True: 
            df = dp.feature_scaling(df)
        if resample == True:
            df = dp.data_resampling(df, timescale)

        # Feature Engineering
        fe = FeatureEngineering(df)
        if engineering == 'time_lag':
            df = fe.add_time_lag_features(df, col_list=[self.target_list], max_lag=1)
        if engineering == 'gain':
            df = fe.transform_gain(df)
        if engineering == 'both':
            df = fe.add_time_lag_features(df, col_list=[self.target_list], max_lag=1)
            df = fe.transform_gain(df)

        if optional == True:
            da = DataAnalysis(df,self.target)
            da.correlation_analysis()
            da.variance_analysis(df)

        # Feature Selection
        fs = FeatureSelection(df, self.target)
        if optional == True:
            if feature_selector == 'dummy':
                selected_tags, selected_df = fs.dummy_feature_importance()
                print(selected_tags)
            if feature_selector == 'boruta':
                selected_tags, selected_df = fs.borutashap_feature_selection()
                print(selected_tags)
            if feature_selector == 'correlation':
                selected_tags, selected_df = fs.correlation_selection()
                print(selected_tags)

        return selected_df