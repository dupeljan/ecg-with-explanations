import pandas as pd
import numpy as np
import wfdb
from wfdb import processing
import ast
import h5py
import matplotlib.pyplot as plot

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = './data/ptb-xl-1.0.1/'
sampling_rate = 500

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y = Y[Y.strat_fold == 5]

Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

X_resample = np.empty((X.shape[0], 4000, X.shape[2]))
# Resample it to 400 Gz
for n in range(X.shape[0]):
    for ch in range(X.shape[2]):
        X_resample[n, :, ch] = processing.resample_sig(X[n, :, ch], 500, 400)[0]

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

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

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Gen test dataset
with h5py.File("ptb_5_fold.hdf5", 'w') as f:
    f.create_dataset('ptb_5', data=X_resample)

# Gen annotation
with open('ptb_5_annotation.csv', 'w') as f:
    data = np.stack(Y.diagnostic_superclass.array)
    data = pd.DataFrame(data=data, columns=['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']).to_csv(index=False)
    f.write(data)
