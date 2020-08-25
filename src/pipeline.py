import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from src.dl_model import DLModel
from src.feature_generator import FeatureGenerator
from src.model import Model

if __name__ == '__main__':
    df = pd.read_csv("data/Anthony_Davis_advanced_all.csv")
    f = FeatureGenerator(df)
    features = f.produce_features()
    m = Model(features)
    # m = DLModel(features)
    m.train()
    m.predict()
    m.eval()