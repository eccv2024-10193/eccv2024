import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt, colors

def get_mean_var(in_tensor):
    mean = np.mean(in_tensor, axis=0)
    var = np.var(in_tensor, axis=0)
    return np.expand_dims(mean, axis=0), np.expand_dims(var, axis=0)


def class_iou(confusion_matrix):
    i = tf.linalg.diag_part(confusion_matrix)
    u = tf.reduce_sum(confusion_matrix, axis=0) + tf.reduce_sum(confusion_matrix, axis=1) - i
    return i / u


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def display(display_list, fn, colormap=None, entropy=None, use_title=None):
    title = ['Input Image', 'True Mask', 'Predicted Mask'] if use_title is None else use_title
    plt_len = len(display_list) if entropy is None else len(display_list) + 1
    f_size = int(display_list[0].shape[0]/100*1.5)
    fig, axs = plt.subplots(plt_len, figsize=(f_size, f_size*len(display_list)))
    for i in range(len(display_list)):
        N = len(np.unique(display_list[i]))
        cmap = colormap[i]
        if colormap[i] is not None:
            cmap = []
            for c in np.unique(display_list[i]):
                cmap.append(colormap[i][int(c)])
        axs[i].set_title(title[i], fontsize=6)
        dimage = (display_list[i] * 255).astype(np.uint8) if i == 0 else display_list[i]
        axs[i].imshow(tf.keras.preprocessing.image.array_to_img(dimage, scale=False),
                      cmap=colors.ListedColormap(cmap, N=N))
        axs[i].axis('off')

    if entropy is not None:
        axs[-1].set_title('Entropy', fontsize=6)
        axs[-1].imshow(tf.keras.preprocessing.image.array_to_img(entropy, scale=True), cmap='gray', vmin=0, vmax=255)
        axs[-1].axis('off')

    fig.tight_layout()
    plt.savefig(fn)
    plt.close("all")


def one_hot(a, ncols=3):
    out = np.zeros((a.size, ncols), dtype=np.uint8)
    out[np.arange(a.size), a.ravel()] = 1
    out.shape = a.shape + (ncols,)
    return out
