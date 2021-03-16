import numpy as np
import processing as proc


def exercise_3_1(img, r, g, b, gamma=None):
    """
    Write a simple application to change the color balance of an image
    by multiplying each color value by a different user-specified constant. If you want to get
    fancy, you can make this application interactive, with sliders.
    TODO: add GUI function to use it with slides
    :param img:
    :param r:
    :param g:
    :param b:
    :return:
    """
    if gamma == 'before':
        img = proc.gamma_correction(img, 2.2)
    img_ret = np.zeros(img.shape)
    img_ret[:, :, 0] = img[:, :, 0] * r
    img_ret[:, :, 1] = img[:, :, 1] * g
    img_ret[:, :, 2] = img[:, :, 2] * b
    img_ret = img_ret.clip(min=0, max=255).astype(np.uint8)
    if gamma == 'after':
        img_ret = proc.gamma_correction(img_ret, 2.2)
    return img_ret