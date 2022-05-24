import random
import numpy as np
import pandas as pd
import catboost as cb
import os
import json
from config import config
from data import get_data, split_data, get_dtypes
import pickle


def predict(X, model_zoo, ths=0.3):
    preds = [model.predict_proba(make_pool(X))[:, 1] >= ths for i, model in enumerate(model_zoo)]
    preds = pd.DataFrame(np.array(preds).transpose(1, 0), index=X.index,
                         columns=[f'type_{i}' for i in range(35)]).astype(int)
    return preds


random.seed(config.seed)
np.random.seed(config.seed)

if __name__ == '__main__':
    print('Reading the data...')
    payments_dtypes, target_dtypes = get_dtypes()

    payments = pd.read_csv(config.paths.payments_test, dtype=payments_dtypes)

    print('Features extraction...')
    features = get_data(payments)

    print('Inference...')
    with open(config.save_path + '2600features_models.pickle', 'rb') as f:
        models = pickle.load(f)

    preds_test = predict(features, models, ths=0.32)

    preds_test.to_csv(config.save_path + config.save_name)
