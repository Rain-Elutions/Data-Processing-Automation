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
    def __init__(self, data_source: str = None, target: str = None, target_list: list = None, problem_type: str = None):
        self.data_source = data_source
        self.target = target
        self.target_list = target_list
        self.problem_type = problem_type

    def pipeline(self):
        # Data Exploration
        print('Loading Data...')
        data_exp = DataExploration()
        df = data_exp.load_data(self.data_source, parse_dates = cfg.pipeline_options.parse_dates, index_col = 0)

        print('Getting Size...')
        data_exp.get_data_size()
        # for i in range(len(df.columns)):
        #     data_exp.get_data_type(df,df.columns[i])

        print('Summarizing Type...')
        data_exp.summarize_data_type()

        print('Summarizing Missing Data...')
        data_exp.summarize_missing_data()

        eda_vis = EDA_Visualization(df)
        eda_vis.visualize_missing_data()
        
        # DataCleansing Module
        data_cleansing = DataCleansing(df)
        print('Removing Duplicates...')
        df = data_cleansing.remove_duplicates(df)

        print('Handling Missing Data...')
        df = data_cleansing.handle_missing_values(df, self.target_list, 
                                                  cfg.pipeline_options.missing_Values.drop_threshold, 
                                                  cfg.pipeline_options.missing_Values.fill_method)
        data_exp.summarize_missing_data(df)

        if cfg.pipeline_options.anomaly == True:
            print('Generating Anomaly Report...')
            data_cleansing.generate_anomaly_report(df, self.target, self.problem_type)
        if cfg.pipeline_options.outliers == True:
            print('Detecting Outliers...')
            for i in range(1,len(df.select_dtypes(include=['number']).columns)):
                data_cleansing.detect_outliers(df.select_dtypes(include=['number']), col_name=df.select_dtypes(include=['number']).columns[i], threshold=3)
       
        # Encoding & Scaling
        dp = DataPreprocessing(df, [self.target_list])
        print('Encoding Features...')
        df = dp.feature_encoding()
        if cfg.pipeline_options.scaling == True: 
            df = dp.feature_scaling(df)

        df = dp.data_resampling(df, cfg.pipeline_options.time_scale)
        
        # keep this?
        # dtypes = df.dtypes.to_dict()
        # for col_name, typ in dtypes.items():
        #     if typ == 'datetime64[ns]': 
        #         df = df.set_index(f'{col_name}')

        # Analysis 
        da = DataAnalysis(df,self.target)
        print('Analyzing Data...')
        # add condtion?
        if True:
            da.correlation_analysis() # too messy
        if df.shape[1] <= 60:
            da.variance_analysis()

        # Feature Selection
        if cfg.pipeline_options.feature_selection.do == True:
            fs = FeatureSelection(df, self.target)
            print('Selecting Features...')
            if cfg.pipeline_options.feature_selection.method == 'dummy':
                selected_tags, df = fs.dummy_feature_importance(
                    select_num = cfg.pipeline_options.feature_selection.select_num,
                    iter = cfg.pipeline_options.feature_selection.iter_num
                )
            if cfg.pipeline_options.feature_selection.method == 'boruta':
                selected_tags, df = fs.borutashap_feature_selection(
                    iter = cfg.pipeline_options.feature_selection.iter_num
                )
            if cfg.pipeline_options.feature_selection.method == 'correlation':
                selected_tags, df = fs.correlation_selection(
                    threshold = cfg.pipeline_options.feature_selection.threshold
                )
            print("selected features: ", selected_tags)

        # Feature Engineering (should this go after selection?)
        # should we make this a seperate df and then concat after feature selection?
        fe = FeatureEngineering(df)
        if cfg.pipeline_options.feature_selection.time_lag == True:
            print('Adding Time Lag Features...')
            df = fe.add_time_lag_features(df, col_list=[self.target_list], max_lag=1)
        if cfg.pipeline_options.feature_selection.time_features == True:
            print('Adding Time Features...')
            df = fe.add_time_features(df)
        if cfg.pipeline_options.feature_selection.gain == True:
            print('Transform the data into gain...')
            df = fe.transform_gain(df)



        return df
