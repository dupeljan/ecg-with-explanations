import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import argparse
import wfdb
import ast
import os

from typing import List
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from wfdb import processing

warnings.filterwarnings("ignore")

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


def get_examples_for_visualization(ptx_large_path=PTB_XL_PATH, fold_n=5):
    sampling_rate = 500

    # load and convert annotation data
    Y = pd.read_csv(ptx_large_path + 'ptbxl_database.csv', index_col='ecg_id')
    Y = Y[Y.strat_fold == fold_n]

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

    X_resampled = np.empty((X.shape[0], 4096, X.shape[2]))
    # Resample it to 400 Gz
    for n in range(X.shape[0]):
        for ch in range(X.shape[2]):
            X_resampled[n, :, ch] = \
                np.concatenate([processing.resample_sig(X[n, :, ch], 500, 400)[0], np.array([0.]*96)])

    return X_resampled, np.stack(Y.diagnostic_superclass.values)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Prepare model: remove last layer's softmax
    model.layers[-1].activation = None
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), tf.argmax(preds[0]).numpy()


def plot_gradcam(x, y_true, y_idx_predicted, heatmap):
    # Load the original image

    # Resize heatmap
    x_points = list(range(heatmap.shape[0]))
    x_points = [xp * (x.shape[0] / x_points[-1]) for xp in x_points]
    heatmap_stretched = np.interp(list(range(x.shape[0])), xp=x_points, fp=heatmap)

    # Plot it
    fig, (ax, *ax_bends) = plt.subplots(nrows=13, sharex=True, **PLOT_OPTIONS)
    fig.suptitle('Diagnose: '+','.join(get_classes_from_logits(y_true)) +\
                 f' Predicted: {map_ad_classes[y_idx_predicted]}', fontsize=40)
    shift = 0.05
    fig.subplots_adjust(shift, shift, 1 - shift, 1 - shift)
    ax.imshow(heatmap_stretched[np.newaxis, :], cmap='Blues',  aspect="auto")
    ax.set_yticks([])
    ax.set_ylabel('Heatmap', fontsize=30)
    for i, ax_bend in enumerate(ax_bends):
        ax_bend.plot(x[:, i])
        ax_bend.grid()
        ax_bend.set_ylabel(bands_names[i], fontsize=30)


def main():
    parser = argparse.ArgumentParser(description='Generate GradCamVisualization for choosen inp')
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

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Import data
    x, y = get_examples_for_visualization()
    # Import model
    model = load_model(args.path_to_model, compile=False)
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    for i, (x_, y_) in enumerate(zip(x, y)):
        heatmap, pred_index = make_gradcam_heatmap(np.expand_dims(x_, axis=0), model,
                                                   'conv1d_11', np.where(y_ == 1.)[0][0])
        plot_gradcam(x_, y_, pred_index, heatmap)
        plt.savefig(os.path.join(OUTPUT_PATH, str(interesting_cases_ecg_id[i]) + '.png'))
        plt.show()
    pass


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    main()
