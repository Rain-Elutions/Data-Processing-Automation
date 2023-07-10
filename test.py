from Data_Exploration.main import DataExploration


if __name__ == '__main__':
    # ----------------- Test DataExploration -----------------
    data_exp = DataExploration()
    # data_exp.load_data('./data/coyote_data_sample.xlsx')
    data_exp.load_data('./data/lng_data_sample.csv')

    
    print("Data size:", data_exp.get_data_size())
    print("Data types:\n", data_exp.get_data_type())
    print("Missing data summary:")
    print(data_exp.summarize_missing_data())
    data_exp.visualize_missing_data()

    # ----------------- Test DataCleansing -----------------
    