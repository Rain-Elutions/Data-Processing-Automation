from dataclasses import dataclass
from typing import Protocol, List
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from box import Box
with open('./config.yaml') as f:
    cfg = Box.from_yaml(f.read())


# can also use ABC instead of Protocol here
class FeatureEncoding(Protocol):
    def get_columns(self) -> list:
        ...

    def encode(self) -> pd.DataFrame:
        ...


class BinaryEncoding:
    def __init__(self, data: pd.DataFrame = None, target_list: List[str] = None):
        self.data = data
        self.target_list = target_list

    def get_columns(self) -> list:
        '''
        Get binary columns from a DataFrame

        Return:
        - binary_cols: a list of binary columns
        '''
        binary_cols = self.data.select_dtypes(include=['bool']).columns.tolist()
        # delete the target column if it is in the list
        for col in self.target_list:
            if col in binary_cols:
                binary_cols.remove(col)

        return binary_cols

    def encode(self) -> pd.DataFrame:
        '''
        Binary encoding is a categorical encoding technique 
        where we map the Boolean values to 0 and 1.

        Parameters:
        - col_list: a list of column names that need to be encoded

        Return:
        - data: a DataFrame with encoded columns
        '''

        col_list = self.get_columns()
        for col in col_list:
            self.data[col] = self.data[col].map({True: 1, False: 0})

        return self.data
    

class TargetEncoding():
    def __init__(self, data: pd.DataFrame = None, target_list: List[str] = None):
        self.data = data
        self.target_list = target_list

    def get_columns(self) -> list:
        '''
        Get categorical columns from a DataFrame

        Return:
        - categorical_cols: a list of categorical columns
        '''

        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        # delete the target column if it is in the list
        for col in self.target_list:
            if col in categorical_cols:
                categorical_cols.remove(col)

        return categorical_cols
    
    def encode(self) -> pd.DataFrame:
        '''
        Target encoding for categorical columns

        Return:
        - data: a DataFrame with categorical columns encoded by target encoding
        '''

        categorical_cols = self.get_columns()

        data_preprocessing = DataPreprocessing()
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing.data_splitting(self.data, self.target_list)

        encoder = TargetEncoder()
        encoder.fit(X_train[categorical_cols], y_train)
        X_train[categorical_cols] = encoder.transform(X_train[categorical_cols])
        X_val[categorical_cols] = encoder.transform(X_val[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

        # Create DataFrames for training, validation, and test sets
        df_train = pd.concat([X_train, y_train], axis=1)
        df_val = pd.concat([X_val, y_val], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        # Concatenate training, validation, and test sets back to the original DataFrame
        self.data = pd.concat([df_train, df_val, df_test])

        return self.data


class OrdinalEncoding():
    def __init__(self, data: pd.DataFrame = None, target_list: List[str] = None):
        self.data = data
        self.target_list = target_list

    def get_columns(self) -> list:
        '''
        Get categorical columns from a DataFrame

        Return:
        - categorical_cols: a list of categorical columns
        '''

        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        # delete the target column if it is in the list
        for col in self.target_list:
            if col in categorical_cols:
                categorical_cols.remove(col)

        return categorical_cols
    
    def encode(self) -> pd.DataFrame:
        '''
        Ordinal encoding is a categorical encoding technique that 
        assigns each unique category in a categorical variable with 
        an integer, based on its alphabetic order.

        Parameters:
        - col_list: a list of column names that need to be encoded

        Return:
        - data: a DataFrame with encoded columns
        '''

        # using .cat.codes
        # for col in col_list:
        #     data[col] = data[col].astype('category')
        #     data[col] = data[col].cat.codes

        # using OrdinalEncoder
        encoder = OrdinalEncoder()
        col_list = self.get_columns()
        self.data[col_list] = encoder.fit_transform(self.data[col_list])

        return self.data


class FeatureScaling():
    def __init__(self, data: pd.DataFrame = None, target_list: List[str] = None):
        self.data = data
        self.target_list = target_list

    def get_columns(self) -> list:
        '''
        Get numerical columns from a DataFrame

        Return:
        - numerical_cols: a list of numerical columns
        '''

        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # delete the target column if it is in the list
        for col in self.target_list:
            if col in numerical_cols:
                numerical_cols.remove(col)

        return numerical_cols

    def scale(self, method: str="minmax") -> pd.DataFrame:
        '''
        Min-max or standard scaling for numerical columns

        Parameters:
        - method: the scaling method, minmax or standard

        Return:
        - data: a DataFrame with numerical columns scaled by min-max scaling or standard scaling
        '''

        numerical_columns = self.get_columns()

        data_preprocessing = DataPreprocessing()
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing.data_splitting(self.data, target_list=self.target_list)

        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid method! Choose from: minmax, standard")

        scaler.fit(X_train[numerical_columns])
        X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])
        X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

        # merge X and y back together
        X_train[self.target_list] = y_train
        X_val[self.target_list] = y_val
        X_test[self.target_list] = y_test

        # merge X_train, X_val, X_test back together
        self.data = pd.concat([X_train, X_val, X_test])

        return self.data


class DataPreprocessing():
    def __init__(self, data: pd.DataFrame = None, target_list: List[str] = None):
        self.data = data
        self.target_list = target_list

    # using protocol to define the interface
    def feature_encoding(self, data: pd.DataFrame=None, method: FeatureEncoding=TargetEncoding) -> pd.DataFrame:
        '''
        Feature encoding for binary and categorical columns (target encoding by default)

        Parameters:
        - data: the input DataFrame

        Return:
        - data: the DataFrame with encoded columns
        '''

        data = data if data is not None else self.data

        # binary encoding
        be = BinaryEncoding(data, self.target_list)
        data = be.encode()
        
        method = TargetEncoding if len(self.target_list) == 1 else OrdinalEncoding
        encoder = method(data, self.target_list)
        data = encoder.encode()
        
        return data

    def feature_scaling(self, data: pd.DataFrame=None) -> pd.DataFrame:
        '''
        Feature scaling for numerical columns (min-max scaling by default)

        Parameters:
        - data: the input DataFrame

        Return:
        - data: the DataFrame with scaled numerical columns
        '''

        data = data if data is not None else self.data

        minmax_scaling = FeatureScaling(data, self.target_list)
        data = minmax_scaling.scale(method="minmax")

        return data

    def data_resampling(self, data: pd.DataFrame=None, frequency: str='') -> pd.DataFrame:
        '''
        Data resampling for time series data

        Parameters:
        - data: the input DataFrame
        - frequency: the frequency of resampling (hourly, daily, weekly, monthly)

        Return:
        - data: the resampled DataFrame
        '''

        data = data if data is not None else self.data

        if frequency == 'hour' or 'Hour' or 'h' or 'H':
            data = data.resample('H').mean()
        elif frequency == 'day' or 'Day' or 'd' or 'D':
            data = data.resample('D').mean()
        elif frequency == 'week' or 'Week' or 'w' or 'W':
            data = data.resample('W').mean()
        elif frequency == 'month' or 'Month' or 'm' or 'M':
            data = data.resample('M').mean()
        else:
            print('Invalid frequency!')

        return data


    def data_splitting(self, data: pd.DataFrame=None, target_list: List[str]=[]) -> pd.DataFrame:
        '''
        Split the data into X/y, training/validation/test sets 

        Parameters:
        - data: the input DataFrame
        - target_list: the list of target columns

        Return:
        - X_train, X_val, X_test, y_train, y_val, y_test, as dataframes
        '''
        test_size, other_cfg = cfg.data_split.test_size, cfg.data_split.other_config

        if other_cfg.shuffle is False:
            if other_cfg.stratify is not None:
                raise ValueError(
                    "Stratified train/test split is not implemented for shuffle=False"
                )
            
        data = data if data is not None else self.data
        target_list = target_list if target_list != [] else self.target_list
        
        X = data.drop(target_list, axis=1)
        y = data[target_list]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size*2, **other_cfg)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, **other_cfg)

        return X_train, X_val, X_test, y_train, y_val, y_test




# df = pd.read_csv('data/raw_data/lng_alldata_1000.csv', parse_dates=True, index_col=0)
# # te = TargetEncoding(df, '3GT1401_3:314FT010.PNT')
# # df = te.encode()
# dp = DataPreprocessing(df, ['3GT1401_3:314FT010.PNT', df.columns[-1]])
# df = dp.feature_encoding(df)
# print(df)