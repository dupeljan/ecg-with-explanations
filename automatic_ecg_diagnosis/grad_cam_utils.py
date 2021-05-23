import wfdb
import ast
import pandas as pd
import numpy as np

from wfdb import processing
from typing import List

PLOT_OPTIONS = {'figsize': (30, 30), 'dpi': 100}

PTB_XL_PATH = '../data/ptb-xl-1.0.1/'
OUTPUT_PATH = '../explanations/ptx-xl/'

map_ad_classes = [
    '1AVB',
    'CRBBB',
    'CLBBB',
    'SBRAD',
    'AFIB',
    'STACH',
]

bands_names = [
    'DI',
    'DII',
    'DIII',
    'AVR',
    'AVL',
    'AVF',
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6'
]

interesting_cases_ecg_id = [
    282,
    424,
    489,
    21563,
    21585,
    1694,
    19715
]


def get_classes_from_logits(logits: np.array) -> List[str]:
    return [map_ad_classes[idx] for idx in np.where(logits == 1.)[0]]


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def get_examples_for_visualization(ptx_large_path=PTB_XL_PATH, resample=True):
    sampling_rate = 500

    # load and convert annotation data
    Y = pd.read_csv(ptx_large_path + 'ptbxl_database.csv', index_col='ecg_id')

    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    def aggregate_diagnostic(y_dic):
        one_hot = np.zeros((len(map_ad_classes)))
        for key in y_dic.keys():
            if key in map_ad_classes:
                one_hot[map_ad_classes.index(key)] = 1.
        return one_hot

    # Apply diagnostic superclass
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    # Choose interesting cases
    #Y = Y[Y['diagnostic_superclass'].apply(lambda x: x[0] == 1)]
    Y = Y.loc[interesting_cases_ecg_id]
    if isinstance(Y, pd.Series):
        Y.diagnostic_superclass = np.array([Y.diagnostic_superclass])
        _ = pd.DataFrame(columns=Y.index)
        _.loc[0] = Y
        Y = _

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, ptx_large_path)

    if not resample:
        X_resampled = X
    else:
        X_resampled = np.empty((X.shape[0], 4096, X.shape[2]))
        # Resample it to 400 Gz
        for n in range(X.shape[0]):
            for ch in range(X.shape[2]):
                X_resampled[n, :, ch] = \
                    np.concatenate([processing.resample_sig(X[n, :, ch], 500, 400)[0], np.array([0.]*96)])

    return X_resampled, np.stack(Y.diagnostic_superclass.values)
