import pandas as pd

class FeatureSelection:
    def __init__(self, data,target_name):
        self.data = data
        self.target_name = target_name

    def correlation_selection(self, threshold: float=.5):
        corr_matrix = self.data.corr()
        # get columns with correlation to target greater than threshold
        target_corr = corr_matrix[self.target_name]
        target_corr_positive = target_corr[target_corr > threshold]
        target_corr_negative = target_corr[target_corr < -threshold]
        high_cor_cols = pd.concat([target_corr_positive, target_corr_negative])

        return high_cor_cols