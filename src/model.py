import pandas as pd
from joblib import load, dump
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Model:

    def __init__(self, df_features: pd.DataFrame):
        self.model = None
        # self.reg = RandomForestRegressor(n_estimators=500)
        self.reg = linear_model.Ridge(alpha=10)
        # self.reg = linear_model.LinearRegression()
        self.features = df_features
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.preds = None

    def split_data(self):
        X = self.features.drop(columns=["target_score"])
        y = self.features["target_score"]
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        pca = PCA(n_components='mle')
        pca.fit(X)
        X = pca.transform(X)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    def train(self):
        self.split_data()
        self.model = self.reg.fit(self.x_train, self.y_train)

        return self.model

    def save(self, path: str):
        dump(self.model, path)

    def load(self, path: str):
        self.model = load(path)

        return self.model

    def predict(self):
        self.split_data()
        self.preds = self.model.predict(self.x_test)

        return self.preds

    def eval(self):
        print("Mean squared error: {}".format(mean_squared_error(self.y_test, self.preds)))
        print("Mean absolute error: {}".format(mean_absolute_error(self.y_test, self.preds)))
        print("r2: {}".format(r2_score(self.y_test, self.preds)))
