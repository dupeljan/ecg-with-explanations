from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import torch
import wfdb
import ast
import matplotlib.pyplot as plot


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


map_ad_classes = [
        '1AVB',
        'CRBBB',
        'CLBBB',
        'SBRAD',
        'AFIB',
        'STACH',
    ]


def aggregate_diagnostic(y_dic):
    one_hot = np.zeros((len(map_ad_classes)))
    for key in y_dic.keys():
        if key in map_ad_classes:
            one_hot[map_ad_classes.index(key)] = 1.
    return one_hot


def load_tenor_dataset(device):
    """Load ptb-xl-1.0.1 dataset as tensor dataset

    :return: tuple of train and test datasets."""
    path = '../data/ptb-xl-1.0.1/'
    sampling_rate = 500

    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')

    # Preprocess
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # filter
    Y = Y[Y.scp_codes.apply(lambda x: any([y in map_ad_classes for y in x]))]
    Y_train = Y[Y.strat_fold != 9]
    Y_test = Y[Y.strat_fold == 9]

    # Load raw signal data
    x_train = load_raw_data(Y_train, sampling_rate, path)
    y_train = np.stack(Y_train.diagnostic_superclass.to_numpy())
    x_test = load_raw_data(Y_test, sampling_rate, path)
    y_test = np.stack(Y_test.diagnostic_superclass.to_numpy())

    x_train, y_train, x_test, y_test = map(lambda t: t.to(device), map(
        torch.tensor, (x_train, y_train, x_test, y_test)
    ))
    x_train = x_train.transpose(1, 2)
    x_test = x_test.transpose(1, 2)
    train_ds = TensorDataset(x_train.double(), y_train.double())
    test_ds = TensorDataset(x_test.double(), y_test.double())

    return train_ds, test_ds


if __name__ == '__main__':
    load_tenor_dataset()
