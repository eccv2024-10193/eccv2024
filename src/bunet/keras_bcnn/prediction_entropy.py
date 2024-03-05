import numpy as np
from scipy.stats import entropy


def pixel_entropy(pixel, base=2):
    """
    Calculate the pixel entropy. The pixel has C channels which are the softmax output
    Args:
        pixel: pixels from softmax
        base:

    Returns: pixel entropy

    """
    return entropy(pixel, base=base)


def image_entropy(img, base=2):
    """
    Calculate the entropy for each pixel of a predicted image.
    Args:
        img: image of C predicted channels from softmax output
        base:

    Returns: pixel-wise entropy

    """
    axis = -1
    pk = 1.0 * img / np.sum(img, axis=-1, keepdims=True)
    with np.errstate(divide='ignore'):
        lg = np.log(pk)
        lg[np.isneginf(lg)] = 0
        vec = pk * lg
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    y = -S
    return y


def image_mean_entropy(np_image):
    i_entropy = image_entropy(np_image)
    return np.mean(i_entropy)


def samples_entropy(sample_images):
    """
    calculate the pixel-wise entropy for each image in a sample and average over the pixels
    Args:
        sample_images: sampled predictions [N, H, W, C]

    Returns: average pixel-wise entropy [H, W]

    """
    samples = np.array([image_entropy(x) for x in sample_images])
    for i, e in enumerate(samples):
        if np.sum(np.isnan(e)) != 0:
            print(i)
            print(sample_images[np.isnan(e)])

    ave = np.mean(samples, axis=0)
    return ave


def main():
    # a = np.array([[[[0.1, 0.1, 0.8], [0.1, 0.3, 0.6], [0.33, 0.33, 0.33]],
    #                [[0.1, 0.3, 0.6], [0.2, 0.5, 0.3], [0.33, 0.33, 0.33]]],
    #               [[[0.1, 0.1, 0.8], [0.1, 0.3, 0.6], [0.33, 0.33, 0.33]],
    #                [[0.1, 0.3, 0.6], [0.2, 0.5, 0.3], [0.33, 0.33, 0.33]]]]
    #              )
    # a=np.expand_dims(a,axis=0)
    # print(a.shape)
    # e = samples_entropy(a)
    # print(e.shape)
    # print(e)
    #
    # a = np.arange(12).reshape(4, 3)
    # print(a)
    # print(np.mean(a))

    b = np.random.rand(20, 512, 1024, 3)
    # b = np.random.rand(51, 102, 3)
    import time
    start = time.time()
    e0 = samples_entropy(b)
    t0 = time.time() - start
    # start = time.time()
    # e1 = full_image_entropy(b)
    # t1 = time.time() - start
    print(t0)


if __name__ == "__main__":
    main()
