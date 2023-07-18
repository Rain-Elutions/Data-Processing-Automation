from Data_Exploration.main import DataExploration


if __name__ == '__main__':
    # ----------------- Test DataExploration -----------------
    data_exp = DataExploration()
    # df = data_exp.load_data('./data/coyote_data_sample.xlsx')
    # df = data_exp.load_data('./data/lng_data_sample.csv')
    df = data_exp.load_data('./data/sasol_data_sample.csv')
    # df = data_exp.load_data(1)


    
    print("Data size:", data_exp.get_data_size())
    print("Data type of specified column:\n", data_exp.get_data_type(df.columns[0]))
    print("Data types summary:\n", data_exp.summarize_data_type())
    print("Missing data summary:")
    print(data_exp.summarize_missing_data())
    data_exp.visualize_missing_data()

    # ----------------- Test DataCleansing -----------------
    