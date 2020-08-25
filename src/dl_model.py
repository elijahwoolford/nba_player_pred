import numpy as np
import pandas as pd
from joblib import dump, load
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.feature_generator import FeatureGenerator

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

look_back = 3
batch_size = 1

class DLModel:

    def __init__(self, df_features: pd.DataFrame):
        self.model = None
        self.features = df_features
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.preds = None


    def transform_data(self):
        scaler = MinMaxScaler(feature_range=(0, 1))
        # X = self.features.drop(columns=["target_score"])
        # y = self.features["target_score"]
        dataset = np.array(self.features["target_score"])
        dataset = scaler.fit_transform(dataset.reshape(-1, 1))

        train_size = int(len(dataset) * 0.7)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        trainX, trainY = self.create_dataset(train, look_back)
        testX, testY = self.create_dataset(test, look_back)

        self.x_train = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        self.x_test = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
        self.y_train = trainY
        self.y_test = testY

        # X = scaler.fit_transform(dataset)
        # pca = PCA(n_components='mle')
        # X = pca.fit_transform(X)
        # self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        # self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], 1, self.x_train.shape[1]))
        # self.x_test = np.reshape(self.x_test, (self.x_test.shape[0], 1, self.x_test.shape[1]))

        return scaler

    def create_dataset(self, dataset, look_back=1):
        X, y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            y.append(dataset[i + look_back, 0])

        return np.array(X), np.array(y)

    def train(self):
        self.transform_data()
        model = Sequential()
        model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        for i in range(100):
            model.fit(self.x_train, self.y_train, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
            model.reset_states()

        self.model = model
        self.save("models/lstm_reg")

        return self.model

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = load(path)

        return self.model

    def predict(self):
        scaler = self.transform_data()
        self.preds = self.model.predict(self.x_test)

        return scaler

    def eval(self):
        scaler = self.transform_data()
        trainPredict = self.model.predict(self.x_train, batch_size=batch_size)
        self.model.reset_states()
        testPredict = self.model.predict(self.x_test, batch_size=batch_size)
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([self.y_train])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([self.y_test])
        # calculate root mean squared error
        # trainScore = mean_squared_error(trainY[0], trainPredict[:, 0])
        # print('Train Score: %.2f RMSE' % (trainScore))
        # testScore = mean_squared_error(testY[0], testPredict[:, 0])
        # print('Test Score: %.2f RMSE' % (testScore))
        # shift train predictions for plotting
        print("Mean squared error: {}".format(mean_squared_error(testY[0], testPredict[:, 0])))
        print("Mean absolute error: {}".format(mean_absolute_error(testY[0], testPredict[:, 0])))
        print("r2: {}".format(r2_score(testY[0], testPredict[:, 0])))

        # print("Mean squared error: {}".format(mean_squared_error(self.y_test, self.preds)))
        # print("Mean absolute error: {}".format(mean_absolute_error(self.y_test, self.preds)))
        # print("r2: {}".format(r2_score(self.y_test, self.preds)))


if __name__ == '__main__':
    df = pd.read_csv("data/Lebron_James_advanced_all.csv")
    f = FeatureGenerator(df)
    features = f.produce_features()
    # m = Model(features)
    m = DLModel(features)
