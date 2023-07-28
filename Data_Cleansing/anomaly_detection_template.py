


class AnomalyDetection:
    def __init__(self, df, target_name, threshold=3, problem_type='max'):
        self.df = df
        self.target_name = target_name
        self.threshold = threshold
        self.problem_type = problem_type
        self.manual_input = None

    def get_bound(self): 
        if self.manual_input:
            return self.manual_input
        else:
            lower_bound, upper_bound = 0, 0
            return [lower_bound, upper_bound]
    
    def get_anomalies(self, lower_bound, upper_bound):
        anomalies = self.df[(self.df[self.target_name] > lower_bound) & (self.df[self.target_name] < upper_bound)]
        return anomalies
    
    # main function
    def generate_anomaly_report(self):
        [lower_bound, upper_bound] = self.get_bound()
        anomalies = self.get_anomalies(lower_bound, upper_bound)
        pass