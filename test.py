from Data_Exploration.main import DataExploration
from Data_Visualization.eda import EDA_Visualization


if __name__ == '__main__':
    # ----------------- Test DataExploration -----------------
    data_exp = DataExploration()

    #### load_data() function ####
    # df = data_exp.load_data('./data/lng_data_sample.csv', parse_dates=True, index_col=0)
    df = data_exp.load_data('./data/sasol_data_sample.csv', parse_dates=True, index_col=0)
    # df = data_exp.load_data('./data/coyote_data_sample.xlsx', parse_dates=False, index_col=0)
    # df = data_exp.load_data('./data/coyote_data_sample_1000.pickle', parse_dates=False, index_col=0)
    # df = data_exp.load_data(1)

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
    