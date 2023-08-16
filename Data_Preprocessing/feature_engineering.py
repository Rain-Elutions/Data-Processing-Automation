import pandas as pd

class FeatureEngineering:
    def __init__(self, data: pd.DataFrame = None):
        self.data = data

    def add_time_lag_features(self, data: pd.DataFrame, target_name: str, max_lag: int = 1) -> pd.DataFrame:
        '''
        Add time lag features to a DataFrame

        Parameters:
        - data: the input DataFrame
        - max_lag: the max time lag to add

        Return:
        - data: the DataFrame with time lag features added
        '''

        data = data if data is not None else self.data

        for time_lag in range(1, max_lag+1):
            data['time_lag_' + str(time_lag)] = data[target_name].shift(time_lag)

        return data
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Add time features to a DataFrame

        Parameters:
        - data: the input DataFrame

        Return:
        - data: the DataFrame with time features added
        '''

        data = data if data is not None else self.data

        data['year'] = data.index.year
        data['month'] = data.index.month
        data['day'] = data.index.day
        data['day_of_week'] = data.index.dayofweek
        data['day_of_year'] = data.index.dayofyear
        data['week_of_year'] = data.index.weekofyear

        return data