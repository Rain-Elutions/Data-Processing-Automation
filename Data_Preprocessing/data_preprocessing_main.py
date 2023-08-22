from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from box import Box
with open('./config.yaml') as f:
    cfg = Box.from_yaml(f.read())


class FeatureEncoding(ABC):
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        self.data = data
        self.target_name = target_name

    @abstractmethod
    def get_columns(self) -> list:
        pass

    @abstractmethod
    def encode(self) -> pd.DataFrame:
        pass


class TargetEncoding(FeatureEncoding):
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        super().__init__(data, target_name)

    def get_columns(self) -> list:
        '''
        Get categorical columns from a DataFrame

        Return:
        - categorical_cols: a list of categorical columns
        '''

        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        # delete the target column if it is in the list
        if self.target_name in categorical_cols:
            categorical_cols.remove(self.target_name)

        return categorical_cols
    
    def encode(self) -> pd.DataFrame:
        '''
        Target encoding for categorical columns

        Return:
        - data: a DataFrame with categorical columns encoded by target encoding
        '''

        data = self.data
        categorical_cols = self.get_columns()

        data_preprocessing = DataPreprocessing()
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing.data_splitting(data, target_list=[self.target_name])

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
        data = pd.concat([df_train, df_val, df_test])

        return data


class BinaryEncoding(FeatureEncoding):
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        super().__init__(data, target_name)
    
    def get_columns(self) -> list:
        '''
        Get binary columns from a DataFrame

        Return:
        - binary_cols: a list of binary columns
        '''
        binary_cols = self.data.select_dtypes(include=['bool']).columns.tolist()
        # delete the target column if it is in the list
        if self.target_name in binary_cols:
            binary_cols.remove(self.target_name)

        return binary_cols

    def encode(self) -> pd.DataFrame:
        '''
        Binary encoding is a categorical encoding technique 
        where we map the Binary values to 0 and 1.

        Parameters:
        - col_list: a list of column names that need to be encoded

        Return:
        - data: a DataFrame with encoded columns
        '''
        data = self.data
        col_list = self.get_columns()
        for col in col_list:
            data[col] = data[col].map({True: 1, False: 0})

        return data


class OrdinalEncoding(FeatureEncoding):
    def __init__(self, data: pd.DataFrame = None, target_name: str = None, col_list: list = []):
        super().__init__(data, target_name)
        self.col_list = col_list

    def get_columns(self) -> list:
        return self.col_list
    
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
        data = self.data
        # using .cat.codes
        # for col in col_list:
        #     data[col] = data[col].astype('category')
        #     data[col] = data[col].cat.codes

        # using OrdinalEncoder
        encoder = OrdinalEncoder()
        col_list = self.get_columns()
        data[col_list] = encoder.fit_transform(data[col_list])

        return data


class FeatureScaling(ABC):
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        self.data = data
        self.target_name = target_name

    @abstractmethod
    def get_columns(self) -> list:
        pass

    @abstractmethod
    def scale(self) -> pd.DataFrame:
        pass


class MinMaxScaling(FeatureScaling):
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        super().__init__(data, target_name)

    def get_columns(self) -> list:
        '''
        Get numerical columns from a DataFrame

        Return:
        - numerical_cols: a list of numerical columns
        '''

        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # delete the target column if it is in the list
        if self.target_name in numerical_cols:
            numerical_cols.remove(self.target_name)

        return numerical_cols

    def scale(self) -> pd.DataFrame:
        '''
        Min-max scaling for numerical columns

        Return:
        - data: a DataFrame with numerical columns scaled by min-max scaling
        '''

        data = self.data
        numerical_columns = self.get_columns()

        data_preprocessing = DataPreprocessing()
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing.data_splitting(data, target_list=[self.target_name])

        scaler = MinMaxScaler()
        scaler.fit(X_train[numerical_columns])
        X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])
        X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

        # merge X and y back together
        X_train[self.target_name] = y_train
        X_val[self.target_name] = y_val
        X_test[self.target_name] = y_test

        # merge X_train, X_val, X_test back together
        data = pd.concat([X_train, X_val, X_test])

        return data


class StandardScaling(FeatureScaling):
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        super().__init__(data, target_name)
    
    def get_columns(self) -> list:
        '''
        Get numerical columns from a DataFrame

        Return:
        - numerical_cols: a list of numerical columns
        '''

        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # delete the target column if it is in the list
        if self.target_name in numerical_cols:
            numerical_cols.remove(self.target_name)

        return numerical_cols
    
    def scale(self) -> pd.DataFrame:
        '''
        Standard scaling for numerical columns

        Return:
        - data: a DataFrame with numerical columns scaled by standard scaling
        '''

        data = self.data
        numerical_columns = self.get_columns()

        data_preprocessing = DataPreprocessing()
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing.data_splitting(data, target_list=[self.target_name])

        scaler = StandardScaler()
        scaler.fit(X_train[numerical_columns])
        X_train[numerical_columns] = scaler.transform(X_train[numerical_columns])
        X_val[numerical_columns] = scaler.transform(X_val[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

        # merge X and y back together
        X_train[self.target_name] = y_train
        X_val[self.target_name] = y_val
        X_test[self.target_name] = y_test

        # merge X_train, X_val, X_test back together
        data = pd.concat([X_train, X_val, X_test])

        return data
    

class DataPreprocessing:
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        self.data = data
        self.target_name = target_name

    def feature_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Feature encoding for binary and categorical columns (target encoding by default)

        Parameters:
        - data: the input DataFrame

        Return:
        - data: the DataFrame with encoded columns
        '''

        data = data if data is not None else self.data

        # binary encoding
        be = BinaryEncoding(data, self.target_name)
        data = be.encode()
        
        # target encoding
        te = TargetEncoding(data, self.target_name)
        data = te.encode()

        return data

    def feature_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Feature scaling for numerical columns (min-max scaling by default)

        Parameters:
        - data: the input DataFrame

        Return:
        - data: the DataFrame with scaled numerical columns
        '''

        data = data if data is not None else self.data

        minmax_scaling = MinMaxScaling(data, self.target_name)
        data = minmax_scaling.scale()

        return data

    def data_resampling(self, data: pd.DataFrame, frequency: str) -> pd.DataFrame:
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

    def data_splitting(self, data: pd.DataFrame=None, target_list: list=[]) -> pd.DataFrame:
        '''
        Split the data into X/y, training/validation/test sets 

        Parameters:
        - data: the input DataFrame
        - target_list: the list of target columns

        Return:
        - X_train, X_val, X_test, y_train, y_val, y_test, as dataframes, if val is True
        - X_train, X_test, y_train, y_test, as dataframes, if val is False
        '''
        test_size, other_cfg = cfg.data_split.test_size, cfg.data_split.other_config

        if other_cfg.shuffle is False:
            if other_cfg.stratify is not None:
                raise ValueError(
                    "Stratified train/test split is not implemented for shuffle=False"
                )
            
        data = data if data is not None else self.data

        X = data.drop(target_list, axis=1)
        y = data[target_list]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size*2, **other_cfg)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, **other_cfg)

        return X_train, X_val, X_test, y_train, y_val, y_test




# df = pd.read_csv('data/raw_data/lng_alldata_1000.csv', parse_dates=True, index_col=0)
# te = TargetEncoding(df, '3GT1401_3:314FT010.PNT')
# df = te.encode()