import numpy as np


# =================================> Point operators
def bright_contrast(img, a, b):
    """
    Take a basic linear operation to an Image ndarray, which can be described as 'des = a .* img + b'
    :param img: An img array
    :param a: gain(also can be defined as contrast)
    :param b: bias(also can be defined as brightness)
    :return: Image ndarray
    """
    src = np.array(img, dtype=np.float)
    dst = src*a + b
    dst = dst.clip(min=0, max=255)
    dst = dst.astype(np.uint8)
    return dst


def gamma_correction(img, gamma):
    """
    Take a basic non-linear operation gamma correction to an Image ndarray,
    which can be described as 'des = img ^ (1 / gamma)'
    :param img: An img array
    :param gamma: gamma value
    :return:
    """
    # gamma correction needs to normalization
    img_norm = img / 255
    dst = np.power(img_norm, 1.0/gamma)
    dst = dst * 255
    dst = dst.astype(np.uint8)
    return dst


# to be perfected
def linear_blend(img1, img2, alpha):
    assert img1.shape == img2.shape, "The two pictures should have the same size!"
    return bright_contrast(img1, (1-alpha), alpha*img2)


def clip_histogram_(hists, threshold=2.0):
    all_sum = sum(hists)
    threshold_value = all_sum / len(hists) * threshold
    total_extra = sum([h - threshold_value for h in hists if h >= threshold_value])
    mean_extra = total_extra / len(hists)

    clip_hists = np.zeros((len(hists)), dtype=np.int)
    for i in range(len(hists)):
        if hists[i] >= threshold_value:
            clip_hists[i] = int(threshold_value + mean_extra)
        else:
            clip_hists[i] = int(hists[i] + mean_extra)

    return clip_hists


def cdf_mapping(vector, clip):
    N = vector.size
    hI = np.histogram(vector, bins=range(257))[0]
    if clip:
        hI = clip_histogram_(hI)
    cI = np.cumsum(hI, dtype=np.float64)
    cI = cI * 256.0 / N
    cI = cI.astype(np.uint8)
    return cI


def hist_equalization(img, clip=False):
    img_ret = np.zeros(img.shape, dtype=np.uint8)
    if len(img.shape) == 3:
        for channel in range(img.shape[2]):
            in_vector = img[:, :, channel].flatten()
            cI = cdf_mapping(in_vector, clip)
            # 通过numpy的数组索引简化代码并加快运行速度
            img_ret[:, :, channel] = cI[img[:, :, channel]]
    else:
        in_vector = img.flatten()
        cI = cdf_mapping(in_vector, clip)
        # 通过numpy的数组索引简化代码并加快运行速度
        img_ret = cI[img]
    return img_ret


def contrast_limited_AHE(img, block):
    h = img.shape[0]
    w = img.shape[1]
    # 判断是灰度图还是彩色图
    if len(img) >= 3:
        channel = img.shape[2]
    else:
        channel = 1
    # 四舍五入以减小分区的不均匀性
    step_r = int(h / block + 0.5)
    step_c = int(w / block + 0.5)
    hist_map = np.zeros((h, w, 256, channel))
    for i in range(block):
        for j in range(block):
            for c in range(channel):
                vector = img[i*step_r:min((i+1)*step_r, h), j*step_c:min((j+1)*step_c, w), c].flatten()
                N = vector.size
                hist_map[i, j, :, c] = cdf_mapping(vector, True)

    # 建立完图后开始进行均衡化
    img_ret = np.zeros(img.shape, dtype=np.uint8)
    for c in range(channel):
        for x in range(h):
            for y in range(w):
                axis_x = x // step_r
                axis_y = y // step_c
                rela_x = (x - (axis_x+0.5)*step_r)/step_r
                rela_y = (y - (axis_y+0.5)*step_c)/step_c
                # 这里不严谨，因为等于零的时候应该是黑色点，黑色点应该直接映射
                sign_x = 1 if rela_x >= 0 else -1
                sign_y = 1 if rela_y >= 0 else -1
                if (axis_x == 0 and axis_y == 0 and rela_x < 0 and rela_y < 0) or \
                   (axis_x == block-1 and axis_y == 0 and rela_x > 0 and rela_y < 0) or \
                   (axis_x == 0 and axis_y == block-1 and rela_x < 0 and rela_y > 0) or \
                   (axis_x == block-1 and axis_y == block-1 and rela_x > 0 and rela_y > 0):
                    img_ret[x, y, c] = hist_map[axis_x, axis_y, img[x, y, c], c]
                elif (axis_x == 0 and rela_x < 0) or (axis_x == block-1 and rela_x > 0):
                        img_ret[x, y, c] = hist_map[axis_x, axis_y, img[x, y, c], c] * (1 - abs(rela_y)) + \
                                           hist_map[axis_x, axis_y + sign_y, img[x, y, c], c] * abs(rela_y)
                elif (axis_y == 0 and rela_y < 0) or (axis_y == block-1 and rela_y > 0):
                    img_ret[x, y, c] = hist_map[axis_x, axis_y, img[x, y, c], c] * (1 - abs(rela_x)) + \
                                       hist_map[axis_x + sign_x, axis_y, img[x, y, c], c] * abs(rela_x)
                else:
                    img_ret[x, y, c] = hist_map[axis_x, axis_y, img[x, y, c], c] * (1 - abs(rela_x)) * (1 - abs(rela_y)) + \
                                       hist_map[axis_x + sign_x, axis_y, img[x, y, c], c] * abs(rela_x) * (1 - abs(rela_y)) + \
                                       hist_map[axis_x, axis_y + sign_y, img[x, y, c], c] * (1 - abs(rela_x)) * abs(rela_y) + \
                                       hist_map[axis_x + sign_x, axis_y + sign_y, img[x, y, c], c] * abs(rela_x) * abs(rela_y)
    return img_ret


# =================================> Linear filtering
box_3 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9
box_3_vec = np.array([1, 1, 1])/3
binominal_3 = np.array([[1, 2, 1]])/4
binominal_5 = np.array([1, 4, 6, 4, 1])/16
binominal_5_2 = np.array([1, 4, 6, 4, 1])/8
laplacian = np.array([[-1, -1, -1],
                      [-1, 8, -1],
                      [-1, -1, -1]])
laplacian2 = np.array([[-0.5, -0.5, -0.5],
                      [-0.5, 4, -0.5],
                      [-0.5, -0.5, -0.5]])


def convolve(arr, kernel, padding='full'):
    h, w = arr.shape
    k_h, k_w = kernel.shape
    if padding == 'full':
        arr_h = h - k_h + 1
        arr_w = w - k_w + 1
        arr_in = np.array(arr)
    elif padding == 'same':
        arr_h = h
        arr_w = w
        arr_in = np.zeros((h+k_h-1, w+k_w-1))
        arr_in[k_h//2: h+k_h//2, k_w//2: w+k_w//2] = arr
    else:
        raise Exception("Invalid padding mode!")
    arr_ret = np.zeros((arr_h, arr_w))
    for i in range(arr_h):
        for j in range(arr_w):
            arr_ret[i, j] = (arr_in[i:i+k_h, j:j+k_w] * kernel[:, :]).sum()
    return arr_ret


class Pyramid:
    # TODO：有关图片尺寸错误信息的判断
    def __init__(self, img):
        self.gaussian = []
        self.laplacian = []
        self.gaussian.append(img)
        self.index = 0

    def pyrDown(self):
        img_copy = self.gaussian[self.index]
        img_temp = cv.resize(img_copy, (img_copy.shape[1]+4, img_copy.shape[0]+4))
        img_temp = convolve_sep(img_temp, binominal_5)
        map_scale = np.zeros((img_temp.shape[0]//2, img_temp.shape[1]//2))
        for i in range(map_scale.shape[0]):
            for j in range(map_scale.shape[1]):
                map_scale[i, j] = img_temp[i*2+1, j*2+1]
        self.gaussian.append(map_scale)
        map_gaussian = self.rescale(self.gaussian[self.index+1])
        self.laplacian.append(self.gaussian[self.index] - map_gaussian)
        self.index += 1

    def rescale(self, arr, scale=2):
        arr_ret = np.zeros((arr.shape[0]*scale, arr.shape[1]*scale))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr_ret[scale*i, scale*j] = arr[i, j]
        return convolve_sep(arr_ret, binominal_5_2, 'same')

    def gaussian_img(self, index):
        img_ret = self.gaussian[index]
        for i in range(index):
            img_ret = self.rescale(img_ret)
        return img_ret

    def laplacian_img(self, index):
        img_ret = self.laplacian[index]
        for i in range(index):
            img_ret = self.rescale(img_ret)
        return img_ret

    def clip(self, arr):
        if arr.shape[0] % 2 == 1:
            arr = arr[:-1, :]
        if arr.shape[1] % 2 == 1:
            arr = arr[:, :-1]
        return arr


def byte_layer(arr):
    """
    比特平面分层，可用于图片压缩与图片主信息提取
    :param arr:
    :return:
    """
    ret_list = [arr//128]
    arr_temp = np.zeros((arr.shape))
    for i in range(7):
        arr_temp += ret_list[i] * 2**(7-i)
        ret_list.append((arr-arr_temp)//2**(6-i))
    list.reverse(ret_list)
    return ret_list


def sorted_filter(arr, k_size):
    h, w = arr.shape
    med = k_size**2//2+1
    arr_ret = np.zeros((arr.shape))
    arr_in = np.zeros((h+k_size-1, w+k_size-1))
    arr_in[k_size//2: h+k_size//2, k_size//2: w+k_size//2] = arr
    for i in range(h):
        for j in range(w):
            temp = np.sort(arr_in[i:i + k_size, j:j + k_size].flatten())
            arr_ret[i, j] = temp[med-1]
    return arr_ret

if __name__ == "__main__":
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    k = np.array([[1, 2, 1]])
    ret1 = sorted_filter(arr, 3)
    pass