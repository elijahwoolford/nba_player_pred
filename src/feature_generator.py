import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class FeatureGenerator:

    def __init__(self, df_player: pd.DataFrame):
        self.raw_data = df_player
        self.clean_data = None
        self.features = None
        self.rolling = 3

    def clean_raw_data(self):
        dropped_columns = ["Unnamed: 0", "Unnamed: 5", "Unnamed: 7", "Tm", "Opp", "Date", "Age", "GS", "Rk", "G"]
        df_temp = self.raw_data[self.raw_data["FG"].isnull() == False].drop(columns=dropped_columns).fillna(
            self.raw_data.mean())
        df_temp["MP"] = df_temp["MP"].apply(self.time_to_decimal)
        self.clean_data = df_temp
        self.calculate_gold_score()

        return df_temp

    def calculate_gold_score(self):
        self.clean_data["gold_score"] = (((self.clean_data["FG"] * 2) + (self.clean_data["3P"]) + (
                self.clean_data["TRB"] * 1.2) + (self.clean_data["AST"] * 1.5) + (self.clean_data["STL"] * 3) + (
                                                  self.clean_data["BLK"] * 3)) - (self.clean_data["TOV"]))

    def produce_features(self):
        self.clean_raw_data()
        all_raw_data = self.raw_data[self.raw_data["FG"].isnull() == False]
        temp = pd.DataFrame()
        for column in self.clean_data.columns:
            temp["MA_" + column] = self.clean_data[column].rolling(self.rolling).mean()

        temp["target_score"] = self.clean_data["gold_score"]
        temp["is_home"] = np.where(all_raw_data["Unnamed: 5"] == "@", 0, 1)
        all_raw_data["Date"] = pd.to_datetime(all_raw_data["Date"])
        all_raw_data["days_between"] = all_raw_data["Date"].shift(1) - all_raw_data["Date"]
        all_raw_data["days_between"] = all_raw_data["days_between"].apply(lambda x: x.days)
        temp["is_back_to_back"] = np.where(all_raw_data["days_between"] == -1, 1, 0)
        opp_encoded = pd.get_dummies(all_raw_data["Opp"])
        temp = pd.concat([temp, opp_encoded], axis=1)
        temp = temp[temp["MA_MP"].isnull() == False]

        self.features = temp

        return self.features

    @staticmethod
    def time_to_decimal(mp):
        min_parts = mp.split(":")
        return round((float(min_parts[1]) / 60) + float(min_parts[0]), 2)


if __name__ == '__main__':
    df = pd.read_csv("data/Anthony_Davis_advanced_all.csv")
    f = FeatureGenerator(df)
    features = f.produce_features()
    features.to_csv("data/Anthony_Davis_features.csv")
    print(features)
