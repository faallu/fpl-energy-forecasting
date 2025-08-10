import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, series: pd.Series, train_size=0.8):
        n = int(len(series) * train_size)
        self.train, self.test = series[:n], series[n:]
        self.model = ARIMA(self.train, order=self.order)
        self.fitted_model = self.model.fit()

    def predict(self, steps=None):
        if steps is None:
            steps = len(self.test)
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

    def evaluate(self):
        forecast = self.predict()
        rmse = np.sqrt(mean_squared_error(self.test, forecast))
        return rmse