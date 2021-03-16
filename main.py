from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import visualization as vis
import processing as proc
import application as app
import cv2 as cv


if __name__ == "__main__":
    img = Image.open("./flower.jpg").convert('L')
    img_np = np.array(img)

    img2 = np.array([[[0, 0, 0], [127, 127, 127]], [[255, 255, 255], [0, 255, 255]]])
    # fig = plt.figure()
    # x = np.linspace(0, 1, 100)[1:]
    # y = 1 / (1 + 1 / x)
    # fig.add_subplot(1, 2, 1)
    # plt.plot(x, y)
    # y = 1 / (1 + np.power(128 / x, 50))
    # fig.add_subplot(1, 2, 2)
    # plt.plot(x, y)
    plt.imshow(img2)

    # pyr = proc.Pyramid(img_np)
    # pyr.pyrDown()
    #

    # img_test2 = img_np-proc.convolve(proc.convolve(img_np, proc.binominal_3, padding='same'), proc.binominal_3.T, padding='same')
    # img_test = (img_np + img_test2).clip(min=0, max=255)
    # vis.compare(img_np, img_test, img_test2)

    # ret = proc.byte_layer(img_np)
    # ret.append(img_np)
    # ret.append(ret[7]*128+ret[6]*64)
    # ret.append(ret[7] * 128 + ret[6] * 64+ret[5]*32)
    # ret.append(ret[7] * 128 + ret[6] * 64 + ret[5] * 32+ret[4]*16)
    # vis.grid_show(ret)

    plt.show()
