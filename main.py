from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import visualization as vis
import processing as proc

# Read image information for RGB and Gray
img = Image.open("./hive.jpg")
img_np = np.array(img)
img_g = img.convert('L')
img_g_np = np.array(img_g)

# Choose the no. of figure to showed
fig_th = 6

if __name__ == "__main__":
    if fig_th == 1:
        '''>>>>Fig 1.1<<<<'''
        '''>>>>Fig 3.1<<<<'''
        vis.compare(img_g_np)
    elif fig_th == 2:
        '''>>>>Fig 2.1, Fig 2.2<<<<'''
        img_hist_eq = proc.hist_equalization(img_np)
        img_g_hist_eq = proc.hist_equalization(img_g_np)
        vis.compare(img_np, img_hist_eq)
    elif fig_th == 3:
        '''>>>>Fig 2.3<<<<'''
        in_vector = img_g_np.flatten()
        cI = proc.cdf_mapping(in_vector, False)
        plt.plot(cI)
    elif fig_th == 4:
        '''>>>>Fig 2.5, Fig 2.6<<<<'''
        img_hist_eq = proc.hist_equalization(img_np)
        img_g_hist_eq = proc.hist_equalization(img_g_np)
        img_ahist_eq = proc.contrast_limited_AHE(img_np, 8)
        img_g_ahist_eq = proc.contrast_limited_AHE(img_g_np, 8)
        vis.compare(img_g_np, img_g_hist_eq, img_g_ahist_eq)
    elif fig_th == 5:
        '''>>>>Fig 3.2<<<<'''
        img_th = proc.global_threshold(img_g_np)
        plt.imshow(img_th, cmap='gray')
    elif fig_th == 6:
        '''>>>>Fig 3.3<<<<'''
        img_th1 = proc.global_threshold(img_g_np)
        img_th2 = proc.adaptive_threshold(img_g_np, block=2)
        img_th3 = proc.adaptive_threshold(img_g_np, block=4)
        vis.grid_show([img_g_np, img_th1, img_th2, img_th3])
    plt.show()
