import plotly.graph_objects as go

class EDA_Visualization:
    def __init__(self, data):
        self.data = data

    def visualize_missing_data(self):
        """
        Visualize the missing data

        Returns:
        - the missing data plots
        """
        try:
            if self.data.shape[1] > 200:
                print("Too many features to visualize")
                return
            num_cols_one_plot = 40
            num_plots = self.data.shape[1] // num_cols_one_plot + 1
            for i in range(num_plots):
                start = i * num_cols_one_plot
                end = min((i+1) * num_cols_one_plot, self.data.shape[1])
                # msno.bar(self.data.iloc[:, start:end])
                # plt.show()

                missing_counts = self.data.iloc[:, start:end].isnull().sum()
                # Create a bar plot of missing values using Plotly
                fig = go.Figure(data=[go.Bar(x=missing_counts.index, y=missing_counts)])
                fig.update_layout(
                    title="Missing Data",
                    xaxis_title="Columns",
                    yaxis_title="Missing Count",
                )
                fig.show()
            return
        except Exception as e:
            print("Error:", str(e))