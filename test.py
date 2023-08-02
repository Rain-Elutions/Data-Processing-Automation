from Data_Exploration.data_exploration_main import DataExploration
from Data_Visualization.eda import EDA_Visualization
from Data_Cleansing.anomaly_detection import AnomalyDetection

if __name__ == '__main__':
    # ----------------- Test DataExploration -----------------
    data_exp = DataExploration()

    #### load_data() function ####
    fname = './data/sasol_data_sample.csv'
    # fname = './data/coyote_data_sample_100.xlsx'
    # fname = './data/coyote_data_sample_1000.pickle'
    # fname = './data/lng_data_sample.csv'
    df = data_exp.load_data(fname, parse_dates=True, index_col=0)

    #### get_data_size() function ####
    data_exp.get_data_size()

    #### get_data_type() and summarize_data_type() function ####
    data_exp.get_data_type(column_name=df.columns[0])
    data_exp.summarize_data_type()
    
    #### summarize_missing_data() function ####
    data_exp.summarize_missing_data()

    #### visualize_missing_data() function ####
    eda_vis = EDA_Visualization(df)
    eda_vis.visualize_missing_data()    

    # ----------------- Test DataCleansing -----------------
    anom_detect = AnomalyDetection(data= df,target_name='OXO-5FI696 Augusta',dataname=fname,problem_type='max')
    anom_detect.anomaly_report()