import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import EcgAttention
from train import CHECKPOINT_LOAD_PATH
from automatic_ecg_diagnosis.grad_cam_utils import get_examples_for_visualization
from automatic_ecg_diagnosis.grad_cam_utils import get_classes_from_logits
from automatic_ecg_diagnosis.grad_cam_utils import interesting_cases_ecg_id
from automatic_ecg_diagnosis.grad_cam_utils import bands_names
from automatic_ecg_diagnosis.grad_cam_utils import PLOT_OPTIONS


PTB_XL_PATH = '../data/ptb-xl-1.0.1/'
OUTPUT_PATH = 'explanations/ptx-xl-ecgConvAttentionNet/'


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform=None):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients.append(grad.cpu().detach())

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


def plot_gradcam(x, y_true, y_idx_predicted, heatmap):
    # Load the original image

    # Resize heatmap

    # Plot it
    fig, ax = plt.subplots(nrows=24, sharex=True, **PLOT_OPTIONS)
    fig.suptitle('Diagnose: '+','.join(get_classes_from_logits(y_true.astype(np.int))) + \
                 f' Predicted: {get_classes_from_logits(y_idx_predicted.astype(np.int))}', fontsize=40)
    shift = 0.05
    fig.subplots_adjust(shift, shift, 1 - shift, 1 - shift)

    for i, (hm, ch) in enumerate(zip(ax[:-1:2], ax[1::2])):
        # Plot heatmap
        x_points = list(range(heatmap[i].shape[0]))
        x_points = [xp * (x.shape[0] / x_points[-1]) for xp in x_points]
        heatmap_stretched = np.interp(list(range(x.shape[0])), xp=x_points, fp=heatmap[i])
        hm.imshow(heatmap_stretched[np.newaxis, :], cmap='Blues',  aspect="auto")
        hm.set_yticks([])
        hm.set_ylabel('Heatmap', fontsize=10)

        # Plot band
        ch.plot(x[:, i])
        ch.grid()
        ch.set_ylabel(bands_names[i], fontsize=30)


def get_heatmaps(checkpoint, x_batch, y_true, gpu=False, norm_per_ch=False):
    device = torch.device("cuda") if torch.cuda.is_available() and gpu else torch.device("cpu")
    model = EcgAttention(num_classes=6).to(device).double()
    if checkpoint:
        print(f'Loading from checkpoint {checkpoint}')
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print('Loaded successfully!')
    model.eval()
    target_layer = model.conv4
    a_and_g = ActivationsAndGradients(model, target_layer)
    res_list = []
    for x, y in zip(x_batch, y_true):
        x = torch.tensor(np.expand_dims(x, axis=0)).double().transpose(1, 2)
        output = model(x)
        loss = 0
        for i in np.where(y==1)[0]:
            loss += output[0][i]

        model.zero_grad()
        loss.backward()

        activations = a_and_g.activations[-1].view(12, 5, -1)
        grads = a_and_g.gradients[-1].view(12, 5, -1)
        alpha = grads.mean(dim=2)

        res = torch.zeros(12, 297)
        for inp_ch in range(12):
            for intern_ch in range(5):
                res[inp_ch] += activations[inp_ch][intern_ch] * alpha[inp_ch][intern_ch]


        if norm_per_ch:
            res -= res.min(1, keepdim=True)[0]
            res /= (res.max(1, keepdim=True)[0] + 1e-16)
        else:
            res -= res.min()
            res /= res.max() + 1e-16

        res_list.append(res.cpu().detach().numpy())

    return res_list


if __name__ == '__main__':
    x_batch, y_true = get_examples_for_visualization(resample=False)
    heatmaps = get_heatmaps(CHECKPOINT_LOAD_PATH, x_batch, y_true)
    for i, (h, x, y) in enumerate(zip(heatmaps, x_batch, y_true)):
        plot_gradcam(x, y, y, h)
        plt.savefig(os.path.join('..', OUTPUT_PATH, str(interesting_cases_ecg_id[i]) + '.png'))
        plt.show()
