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
    preds = [model.predict_proba(make_pool(X))[:,1] >= ths for i, model in enumerate(model_zoo)]
    preds = pd.DataFrame(np.array(preds).transpose(1, 0), index=X.index, columns=[f'type_{i}' for i in range(35)]).astype(int)
    return preds


random.seed(config.seed)
np.random.seed(config.seed)

if __name__ == '__main__':
    print('Reading the data...')
    payments_dtypes, target_dtypes = get_dtypes()

    payments = pd.read_csv(config.paths.payments_train, dtype=payments_dtypes)
    target = pd.read_csv(config.paths.target_train, dtype=target_dtypes).set_index('client_id')

    print('Features extraction...')
    features = get_data(payments)

    X_train, y_train, X_val, y_val = split_data(features, target)

    print()
    models = []
    for i in range(35):
        print('Fitting model', i)

        model = cb.CatBoostClassifier(loss_function=config.model.loss_function,
                                      random_seed=config.model.seed,
                                      learning_rate=config.model.learning_rate,
                                      iterations=config.model.iterations,
                                      early_stopping_rounds=config.model.early_stopping_rounds)

        pool_train, pool_val = make_pool(X_train, y_train[f'type_{i}']), make_pool(X_val, y_val[f'type_{i}'])

        model.fit(pool_train, eval_set=pool_val, plot=False, verbose=100)

        models.append(model)

    print('Models trained successfully')

    with open(f'{config.paths.save_path}/{config.name}.pickle', 'wb') as f:
        pickle.dump(models, f)

    print('Models saved successfully')

    print('Validating the models...')
    for ths in np.arange(0.25, 0.35, 0.01):
        preds = predict(X_val, models, ths)
        score = fbeta_score(y_val, preds, beta=0.5, average='micro', zero_division=0)
        print(f'Score for thresh {ths}:', score)















