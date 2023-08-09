import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class FeatureEncoding:
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        self.data = data
        self.target_name = target_name

    def get_boolean_columns(self) -> list:
        '''
        Get boolean columns from a DataFrame

        Return:
        - boolean_cols: a list of boolean columns
        '''
        boolean_cols = self.data.select_dtypes(include=['bool']).columns.tolist()
        # delete the target column if it is in the list
        if self.target_name in boolean_cols:
            boolean_cols.remove(self.target_name)

        return boolean_cols

    def get_categorical_columns(self) -> list:
        categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        # delete the target column if it is in the list
        if self.target_name in categorical_cols:
            categorical_cols.remove(self.target_name)

        return categorical_cols
    
    def target_encoding(self) -> pd.DataFrame:
        data = self.data
        categorical_cols = self.get_categorical_columns()

        data_preprocessing = DataPreprocessing()
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing.data_splitting(data, target_name=self.target_name)

        encoder = TargetEncoder()
        encoder.fit(X_train[categorical_cols], y_train)
        X_train[categorical_cols] = encoder.transform(X_train[categorical_cols])
        X_val[categorical_cols] = encoder.transform(X_val[categorical_cols])
        X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

        # merge X and y back together
        X_train[self.target_name] = y_train
        X_val[self.target_name] = y_val
        X_test[self.target_name] = y_test

        # merge X_train, X_val, X_test back together
        data = pd.concat([X_train, X_val, X_test])

        return data
    
    def boolean_encoding(self, col_list: list) -> pd.DataFrame:
        '''
        Boolean encoding is a categorical encoding technique 
        where we map the Boolean values to 0 and 1.

        Parameters:
        - col_list: a list of column names that need to be encoded

        Return:
        - data: a DataFrame with encoded columns
        '''
        data = self.data
        for col in col_list:
            data[col] = data[col].map({True: 1, False: 0})

        return data

    def ordinal_encoding(self, col_list: list) -> pd.DataFrame:
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
        data[col_list] = encoder.fit_transform(data[col_list])

        return data


class FeatureScaling:
    def __init__(self, data: pd.DataFrame = None, target_name: str = None):
        self.data = data
        self.target_name = target_name

    # apply the same scaling or normalization to validation and test sets that applied to the training set
    def get_numerical_columns(self) -> list:
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # delete the target column if it is in the list
        if self.target_name in numerical_cols:
            numerical_cols.remove(self.target_name)

        return numerical_cols
    
    def min_max_scaling(self) -> pd.DataFrame:
        data = self.data
        numerical_columns = self.get_numerical_columns()

        data_preprocessing = DataPreprocessing()
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing.data_splitting(data, target_name=self.target_name)

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
    
    def standard_scaling(self) -> pd.DataFrame:
        data = self.data
        numerical_columns = self.get_numerical_columns()

        data_preprocessing = DataPreprocessing()
        X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing.data_splitting(data, target_name=self.target_name)

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
    def __init__(self, data: pd.DataFrame = None):
        self.data = data

    def feature_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data if data is not None else self.data

    def feature_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data if data is not None else self.data

    def data_resampling(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data if data is not None else self.data

    def data_splitting(self, data: pd.DataFrame = None, target_name: str = '', test_size: float = 0.1, shuffle: bool = False) -> pd.DataFrame:
        data = data if data is not None else self.data
        X = data.drop(target_name, axis=1)
        y = data[target_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size*2, shuffle=shuffle, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=shuffle, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data if data is not None else self.data


# Test
if __name__ == '__main__':
    # Ordinal encoding
    # data = {
    #     'Temperature': ['Hot', 'Cold', 'Warm', 'Hot', 'Warm', 'Cold'],
    #     'Weather': ['Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Cloudy', 'Rainy'],
    #     'Humidity': ['High', 'Low', 'Medium', 'Medium', 'High', 'Low']
    # }
    # df = pd.DataFrame(data=data)

    # feature_encoding = FeatureEncoding(df)
    # print(feature_encoding.ordinal_encoding(['Temperature', 'Weather', 'Humidity']))

    # Target encoding
    data = {
        "City": ["New York", "Los Angeles", "Chicago", "Houston", "Miami", "Toronto", "Vancouver", "London", "Sydney", "Tokyo"],
        "Country": ["USA", "USA", "USA", "USA", "USA", "Canada", "Canada", "UK", "Australia", "Japan"],
        "Population": [8623000, 3990456, 2716000, 2320268, 4633470, 2731571, 675218, 8982000, 5312163, 13929286]
    }
    df = pd.DataFrame(data)

    feature_encoding = FeatureEncoding(df, target_name='Population')
    df = feature_encoding.target_encoding()
    print(df)
