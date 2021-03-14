import numpy as np
import pandas as pd
import warnings
import argparse
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from datasets import ECGSequence
from sklearn.metrics import classification_report

from generate_figures_and_tables import get_optimal_precision_recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('--path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('--path_to_labels_csv', type=str,
                        help='path to labels csv file.', default='')
    parser.add_argument('--path_to_model',  # or model_date_order.hdf5
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
                        help='output csv file.')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Import data
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.bs)
    # Import model
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    y_score = model.predict(seq,  verbose=1)

    # Generate dataframe
    np.save(args.output_file, y_score)

    print("Output predictions saved")

    if args.path_to_labels_csv != '':
        # Magic
        columns=['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
        threshold = np.array([0.31871256, 0.51558906, 0.5997846, 0.3313214, 0.50154346,
                              0.4270824])
        y_true = pd.read_csv(args.path_to_labels_csv).to_numpy()
        # _, _, threshold = get_optimal_precision_recall(y_true, y_score)
        mask = y_score > threshold
        y_pred = np.zeros_like(y_score)
        y_pred[mask] = 1
        print(classification_report(y_true, y_pred, target_names=columns))
