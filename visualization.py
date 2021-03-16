from matplotlib import pyplot as plt
import math


def onecolor_hist(vec, color):
    x, *_ = plt.hist(vec, bins=range(257), color=color, histtype='step')
    peek = x.argmax()
    return peek


def hist_show(img, ax):
    if len(img.shape) == 3:
        r = img[:, :, 0].flatten()
        g = img[:, :, 1].flatten()
        b = img[:, :, 2].flatten()
        r_peek = onecolor_hist(r, 'r')
        g_peek = onecolor_hist(g, 'g')
        b_peek = onecolor_hist(b, 'b')
        str_argmax = "peek=(%d, %d, %d)" % (r_peek, g_peek, b_peek)
        ax.set_title(str_argmax)
    elif len(img.shape) == 2:
        g = img.flatten()
        g_peek = onecolor_hist(g, 'r')
        str_argmax = "peek=(%d)" % (g_peek)
        ax.set_title(str_argmax)


def compare(*args):
    """
    To show the images in the line 1 compared with histogram in the line 2.
    :param args: The arbitrary number of images(need numpy type)
    :return: Nothing
    """
    img_num = len(args)
    fig = plt.figure()
    for i in range(img_num):
        fig.add_subplot(2, img_num, i+1)
        if len(args[i].shape) == 3:
            plt.imshow(args[i])
        else:
            plt.imshow(args[i], cmap='gray')
        ax = fig.add_subplot(2, img_num, img_num+i+1)
        hist_show(args[i], ax)


def grid_show(show_list):
    show_sum = len(show_list)
    col = math.ceil(math.sqrt(show_sum))
    row = math.ceil(show_sum/col)
    fig = plt.figure()
    for i in range(row):
        for j in range(col):
            if i*col+j+1 > show_sum:
                break
            fig.add_subplot(row, col, i*col+j+1)
            plt.imshow(show_list[i*col+j], cmap='gray')