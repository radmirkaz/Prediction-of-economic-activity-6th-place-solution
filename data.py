import random
import numpy as np
import pandas as pd
import sklearn.utils as sku
from skmultilearn.model_selection import iterative_train_test_split
import os
import json
from config import config

random.seed(config.seed)
np.random.seed(config.seed)


def get_dtypes():
    payments_dtypes = {
        'client_id': str,
        'contractor_id': str,
        'is_outgoing': bool,
        'amount': 'uint64',
        'dt_day': 'uint16',
        'dt_hour': 'uint8',
        'channel': pd.CategoricalDtype()
    }
    for i in range(12):
        payments_dtypes[f'flag_{i}'] = bool

    target_dtypes = {
        'client_id': str
    }
    for i in range(35):
        target_dtypes[f'type_{i}'] = int

    return payments_dtypes, target_dtypes


def get_flags_features(payments):
    flags_features = pd.DataFrame({'client_id': payments['client_id'].unique()})

    for i in range(12):
        subset = pd.pivot_table(payments,
                                index='client_id',
                                values=['amount'],
                                columns=[f'flag_{i}'],
                                aggfunc=['mean', 'median', 'std', 'min', 'max', 'sum', len])

        flags_cols = []
        for name in ['mean', 'median', 'std', 'min', 'max', 'sum', 'len']:
            for bo in ['False', 'True']:
                flags_cols.append(f'flag{i}_{name}:{bo}')
        subset.columns = flags_cols

        flags_features = flags_features.merge(subset, on='client_id', how='left')

    return flags_features.set_index('client_id')


def get_season_features(payments):
    season_features = pd.pivot_table(payments,
                                     index='client_id',
                                     values=['amount'],
                                     columns=['season'],
                                     aggfunc=['mean', 'median', 'std', 'min', 'max', 'sum', len])

    season_cols = []
    for name in ['mean', 'median', 'std', 'min', 'max', 'sum', 'len']:
        for i in range(1, 5):
            season_cols.append(f'season{i}_{name}')
    season_features.columns = season_cols

    return season_features


def get_month_features(payments):
    month_features = pd.pivot_table(payments,
                                    index='client_id',
                                    values=['amount'],
                                    columns=['month'],
                                    aggfunc=['mean', 'median', 'std', 'min', 'max', 'sum', len])

    month_cols = []
    for name in ['mean', 'median', 'std', 'min', 'max', 'sum', 'len']:
        for i in range(1, 13):
            month_cols.append(f'month{i}_{name}')
    month_features.columns = month_cols

    return month_features


def get_month_outgoing_features(payments):
    month_outgoing_features = pd.pivot_table(payments,
                                             index='client_id',
                                             values=['amount'],
                                             columns=['month', 'is_outgoing'],
                                             aggfunc=['mean', 'median', 'std', 'min', 'max', 'sum', len])

    month_outgoing_cols = []
    for name in ['mean', 'median', 'std', 'min', 'max', 'sum', 'len']:
        for month in range(1, 13):
            for bo in ['False', 'True']:
                month_outgoing_cols.append(f'month{month}_outgoing_{name}:{bo}')

    month_outgoing_features.columns = month_outgoing_cols

    return month_outgoing_features


def get_season_channel_features(payments):
    season_channel_features = pd.pivot_table(payments,
                                             index='client_id',
                                             values=['amount'],
                                             columns=['season', 'channel'],
                                             aggfunc=['mean', 'median', 'std', 'min', 'max', 'sum', len])

    season_channel_cols = []
    for name in ['mean', 'median', 'std', 'min', 'max', 'sum', 'len']:
        for season in range(1, 5):
            for bo in ['app', 'atm', 'pos', 'web']:
                season_channel_cols.append(f'season{season}_channel_{name}:{bo}')

    season_channel_features.columns = season_channel_cols

    return season_channel_features


def get_flags_month_features(payments):
    flags_month_features = pd.DataFrame({'client_id': payments['client_id'].unique()})

    for i in range(12):
        subset = pd.pivot_table(payments,
                                index='client_id',
                                values=['amount'],
                                columns=['month', f'flag_{i}'],
                                aggfunc=['mean', 'median', 'std', 'min', 'max', 'sum', len])

        flags_month_cols = []
        for name in ['mean', 'median', 'std', 'min', 'max', 'sum', 'len']:
            for month in range(1, 13):
                for bo in ['False', 'True']:
                    flags_month_cols.append(f'month{month}_flag{i}_{name}:{bo}')
        subset.columns = flags_month_cols

        flags_month_features = flags_month_features.merge(subset, on='client_id', how='left')

    return flags_month_features.set_index('client_id')


def get_flags_season_features(payments):
    flags_season_features = pd.DataFrame({'client_id': payments['client_id'].unique()})

    for i in range(12):
        subset = pd.pivot_table(payments,
                                index='client_id',
                                values=['amount'],
                                columns=['season', f'flag_{i}'],
                                aggfunc=['mean', 'median', 'std', 'min', 'max', 'sum', len])

        flags_season_cols = []
        for name in ['mean', 'median', 'std', 'min', 'max', 'sum', 'len']:
            for season in range(1, 5):
                for bo in ['False', 'True']:
                    flags_season_cols.append(f'season{season}_flag{i}_{name}:{bo}')
        subset.columns = flags_season_cols

        flags_season_features = flags_season_features.merge(subset, on='client_id', how='left')

    return flags_season_features.set_index('client_id')


def get_date_features(payments):
    features = payments.groupby('client_id')['dt_day'].agg(['count', 'nunique', 'min', 'max'])
    features['mean_cnt_trans_per_day'] = features['count'] / features['nunique']
    features['len_period'] = features['max'] - features['min'] + 1
    features['density_trans'] = features['count'] / features['len_period']
    features.columns = ['cnt_tr', 'cnt_tr_days', 'first_tr_day', 'last_tr_day', \
                        'mean_cnt_tr_per_day', 'len_tr_period', 'density_tr']

    return features


def get_outgoing_features(payments):
    features = payments.groupby('client_id')['is_outgoing'].agg(['nunique'])
    features.columns = ['cnt_is_outgoing']

    cnt_outgoing = payments.pivot_table(index='client_id', values=['amount'], columns=['is_outgoing'],
                                   aggfunc=['count']).fillna(0)
    cnt_outgoing.columns = [f'{str(i[0])}-is_outgoing:{str(i[2])}' for i in cnt_outgoing.columns]

    outgoing_features = payments.pivot_table(index='client_id', values=['amount'], columns=['is_outgoing'],
                                       aggfunc=['sum', 'mean']).fillna(0)
    outgoing_features.columns = [f'{str(i[0])}-is_outgoing:{str(i[2])}' for i in outgoing_features.columns]

    X = pd.concat([
        features.reindex(payments['client_id'].values),
        cnt_outgoing.reindex(payments['client_id'].values),
        outgoing_features.reindex(payments['client_id'].values)
    ], axis=1).reset_index()

    X = X.drop_duplicates('client_id')

    return X.set_index('client_id')


def get_hour_features(payments):

    hour_features = payments.pivot_table(index='client_id', values=['amount'], columns=['dt_hour'],
                                             aggfunc=['count']).fillna(0)
    hour_features.columns = [f'{str(i[0])}-hour:{str(i[2])}' for i in hour_features.columns]

    return hour_features


def get_channel_amount_features(payments):
    features = payments.groupby('client_id')['channel'].agg(['nunique'])
    features.columns = ['cnt_channels']

    channel_features = payments.pivot_table(index='client_id', values=['amount'], columns=['channel'],
                                   aggfunc=['count', 'min', 'max']).fillna(0)
    channel_features.columns = [f'{str(i[0])}-channel:{str(i[2])}' for i in channel_features.columns]

    return channel_features


def get_bs_features(payments):
    fts = payments.groupby('client_id')['amount'].agg(['mean', 'median', 'std', 'min', 'max'])
    fts[[f'flag_{i}_count' for i in range(12)]] = payments.groupby('client_id')[[f'flag_{i}' for i in range(12)]].sum()
    return fts


def get_data(payments):
    payments['month'] = payments['dt_day'] // 31

    baseline_features = get_bs_features(payments)
    date_features = get_date_features(payments)
    outgoing_features = get_outgoing_features(payments)
    hour_features = get_hour_features(payments)
    channel_amount_features = get_channel_amount_features(payments)
    flags_features = get_flags_features(payments)
    season_features = get_season_features(payments)
    month_features = get_month_features(payments)
    month_outgoing_features = get_month_outgoing_features(payments)
    season_channel_features = get_season_channel_features(payments)
    flags_month_features = get_flags_month_features(payments)
    # flags_season_features = get_flags_season_features(payments)

    features = pd.merge(baseline_features, date_features, on='client_id', how='left')
    features = features.merge(outgoing_features, on='client_id', how='left')
    features = features.merge(hour_features, on='client_id', how='left')
    features = features.merge(channel_amount_features, on='client_id', how='left')
    features = features.merge(flags_features, on='client_id', how='left')
    features = features.merge(season_features, on='client_id', how='left')
    features = features.merge(month_features, on='client_id', how='left')
    features = features.merge(month_outgoing_features, on='client_id', how='left')
    features = features.merge(season_channel_features, on='client_id', how='left')
    features = features.merge(flags_month_features, on='client_id', how='left')
    # features = features.merge(flags_season_features, on='client_id', how='left')
    print(features.shape)

    return features


def stratified_split_cached(X, y, split_idx_file):
    if os.path.isfile(split_idx_file):
        with open(split_idx_file, 'r') as f:
            split_json = json.load(f)
        train_idx, val_idx = split_json['train'], split_json['val']
    else:
        y_shuffle = sku.shuffle(y, random_state=config.seed)  # https://cpb-us-e1.wpmucdn.com/journeys.dartmouth.edu/dist/8/830/files/2020/06/EIqwWwsX0AAeh-o.jpeg
        train_idx, _, val_idx, _ = iterative_train_test_split(np.expand_dims(y_shuffle.index, 1), np.array(y_shuffle), test_size=0.15)
        train_idx, val_idx = train_idx.squeeze(1), val_idx.squeeze(1)
        with open(split_idx_file, 'w') as f:
            json.dump({'train': list(train_idx), 'val': list(val_idx)}, f)
    return X.loc[train_idx], y.loc[train_idx], X.loc[val_idx], y.loc[val_idx]


def split_data(features, target):
    X_train, y_train, X_val, y_val = stratified_split_cached(features, target, config.data.split_file_path)
    print('Train size:', X_train.shape, 'Val size:', X_val.shape)

    return X_train, y_train, X_val, y_val









