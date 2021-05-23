import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from grad_cam_utils import *

warnings.filterwarnings("ignore")



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
