import pandas as pd
from box import Box
with open('config.yaml') as f:
    cfg = Box.from_yaml(f.read())
import warnings
warnings.filterwarnings('ignore')
from Data_Exploration.data_exploration_main import DataExploration
from Data_Cleansing.data_cleansing_main import DataCleansing
from Data_Preprocessing.data_preprocessing_main import DataPreprocessing, TargetEncoding, BinaryEncoding, OrdinalEncoding, FeatureScaling
from Data_Preprocessing.feature_engineering import FeatureEngineering
from Data_Analyzing.data_analysis_main import DataAnalysis
from Data_Analyzing.feature_selection import FeatureSelection
from Data_Visualization.eda import EDA_Visualization

class DataProcessing:
    def __init__(self, data_source: str=None, target: str = None):
        self.data_source = data_source
        self.target = target

    def pipeline(self):
        data_exp = DataExploration()

        df = data_exp.load_data(self.data_source, parse_dates=False, index_col=0)
        data_exp.get_data_size()
        data_exp.get_data_type(df,df.columns[0])
        data_exp.summarize_data_type()
        data_exp.summarize_missing_data()

        eda_vis = EDA_Visualization(df)
        eda_vis.visualize_missing_data()

        # Test DataCleansing Module
        data_cleansing = DataCleansing(df)

        df = data_cleansing.remove_duplicates(df)
        data_exp.summarize_missing_data(df)

        df = data_cleansing.handle_missing_values(df, target_list=[self.target], drop_thresh=0.5, fill_missing_method='mean')
        data_exp.summarize_missing_data(df)

        ##########FIX ME############
        data_cleansing.generate_anomaly_report(df, self.target, problem_type='max')
        data_cleansing.detect_outliers(df, col_name=df.columns[10], threshold=3)

        # encoding & scaling
        dp = DataPreprocessing(df, [self.target])
        df = dp.feature_encoding()
        

        ##########FIX ME############
        df = dp.feature_scaling(df)
        df = dp.data_resampling(df, 'h')

        ##########FIX ME############
        # feature engineering
        fe = FeatureEngineering(df)
        df = fe.add_time_lag_features(df, col_list=[self.target], max_lag=1)
        df = fe.transform_gain(df)

        # Test DataAnalysis Module
        # correlation_analysis
        da = DataAnalysis(df,self.target)
        da.correlation_analysis()

        # variance_analysis
        da.variance_analysis(df)

        # feature_selection
        fs = FeatureSelection(df, self.target)
        ##########FIX ME############
        selected_tags, selected_df = fs.dummy_feature_importance()
        print(selected_tags)

        selected_tags, selected_df = fs.borutashap_feature_selection()
        print(selected_tags)

        selected_tags, selected_df = fs.correlation_selection()
        print(selected_tags)

        return selected_df
