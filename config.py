from omegaconf import OmegaConf

config = {
    'seed': 0xCAFEC0DE,
    'name': 'default',

    'paths': {
        'payments_train': '../data/payments_train.csv',
        'payments_test': '../data/payments_test.csv',

        'target_train': '../data/target_train.csv',
        'client_id_test': '../data/client_id_test.csv',

        'save_path': '../models/',
        'save_name': 'submission_hope_dies_last.csv'
},

    'data': {
        'val_size': 0.15,
        'split_file_path': '../split_cache.json'
    },

    'model': {
        'loss_function': 'LogLoss',
        'learning_rate': 0.05,
        'iterations': 2000,
        'early_stopping_rounds': 400,
        'random_seed': '${seed}',
    }
}

config = OmegaConf.create(config)