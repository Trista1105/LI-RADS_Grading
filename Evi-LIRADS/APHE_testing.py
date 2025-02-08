# calculate APHE score

import array_proc
import numpy as np

import copy
import cv2
import matplotlib.pyplot as plt

import display

# 当前APHE/Washout使用的函数是：APHE_Washout_ZhongShan

def find_min_cv_roi0(arr, roi_size=(20, 20)):
    # 初始化变异系数的最小值为无穷大，用于比较
    min_cv = np.inf
    # 初始化最佳ROI区域为None
    best_roi = None
    best_roi_row = 0
    best_roi_col = 0
    best_roi_mask = np.zeros((arr.shape[0], arr.shape[1]))

    # 获取数组的维度
    rows, cols = arr.shape

    # 遍历可能的起点位置
    for i in range(rows - roi_size[0] + 1):
        for j in range(cols - roi_size[1] + 1):
            # 获取当前ROI
            current_roi = arr[i:i + roi_size[0], j:j + roi_size[1]]

            # 检查ROI中是否所有值都大于0
            if np.all(current_roi > 0):
                # 计算变异系数
                std_dev = np.std(current_roi)
                mean = np.mean(current_roi)

                # 防止平均值为0的情况
                if mean > 0:
                    cv = std_dev / mean
                    # 检查是否为最小CV
                    if cv < min_cv:
                        min_cv = cv
                        best_roi = current_roi
                        best_roi_row = i
                        best_roi_col = j

    best_roi_mask[best_roi_row:best_roi_row + roi_size[0], best_roi_col:best_roi_col + roi_size[1]] = 1

    return best_roi, best_roi_mask


from scipy.ndimage import uniform_filter

def find_min_cv_roi1(arr, roi_size):
    best_roi = None
    best_roi_mask = np.zeros_like(arr)

    rows, cols = arr.shape
    roi_h, roi_w = roi_size

    # 将输入数组转换为浮点数，防止整数溢出
    arr = arr.astype(float)  # 从 float32 转成了 float64

    # 计算局部均值
    mean_arr = uniform_filter(arr, size=roi_size, mode='constant', cval=0)

    # 计算局部平方均值
    mean_arr_squared = uniform_filter(arr ** 2, size=roi_size, mode='constant', cval=0)

    # 计算局部方差
    variance = mean_arr_squared - mean_arr ** 2

    # 计算局部标准差
    std_dev = np.sqrt(variance)

    # 提取窗口完全在输入矩阵范围内那些位置的结果
    arr_shape = np.array(arr.shape)
    roi_size = np.array(roi_size)
    half_roi = roi_size // 2
    valid_start = half_roi
    valid_end = arr_shape - (roi_size - 1) // 2

    # 构建有效区域的切片
    valid_slices = tuple(slice(start, end) for start, end in zip(valid_start, valid_end))

    # 提取有效位置的均值和标准差
    mean_arr_window_within_input = mean_arr[valid_slices]
    std_dev_window_within_input = std_dev[valid_slices]

    # 找出均值非 NA（窗口中所有元素 > 0）且均值不为零的位置
    valid_mask = (~np.isnan(mean_arr_window_within_input)) & (mean_arr_window_within_input > 0)

    # 计算变异系数（CV）
    cv = np.full_like(mean_arr_window_within_input, fill_value=np.nan, dtype=float)
    cv[valid_mask] = std_dev_window_within_input[valid_mask] / mean_arr_window_within_input[valid_mask]

    # 排除所有元素都是 NA 的情况
    if not np.all(np.isnan(cv)):
        min_index_linear = np.nanargmin(cv)  # 注意排除 NA 的影响
        min_index_2d = np.unravel_index(min_index_linear, cv.shape)
        best_roi_row, best_roi_col = min_index_2d

        # 获取最佳 ROI
        best_roi = arr[best_roi_row:best_roi_row + roi_h, best_roi_col:best_roi_col + roi_w]
        best_roi_mask[best_roi_row:best_roi_row + roi_h, best_roi_col:best_roi_col + roi_w] = 1

    return best_roi, best_roi_mask


def roi_all_larger_than_zero(arr, roi_size=(20, 20)):
    # 获取数组的维度
    rows, cols = arr.shape
    roi_h, roi_w = roi_size

    # 确保输入的 roi_size 合法
    if roi_h > rows or roi_w > cols:
        raise ValueError("ROI size must be smaller than the array dimensions.")

    # 创建有效性数组
    valid_position = np.zeros((rows - roi_h + 1, cols - roi_w + 1), dtype=bool)

    # 创建一个大于0的布尔数组
    greater_than_zero = arr > 0

    # 计算滑动窗口内的和
    for i in range(rows - roi_h + 1):
        for j in range(cols - roi_w + 1):
            # 使用切片直接在布尔数组上进行求和
            if np.sum(greater_than_zero[i:i + roi_h, j:j + roi_w]) == roi_h * roi_w:
                valid_position[i, j] = 1

    return valid_position

def find_min_cv_roi(arr, roi_size):
    best_roi = None
    best_roi_mask = np.zeros_like(arr)

    rows, cols = arr.shape
    roi_h, roi_w = roi_size

    # 将输入数组转换为浮点数，防止整数溢出
    arr = arr.astype(float)  # 从 float32 转成了 float64

    # 计算局部均值
    mean_arr = uniform_filter(arr, size=roi_size, mode='constant', cval=0)

    # 计算局部平方均值
    mean_arr_squared = uniform_filter(arr ** 2, size=roi_size, mode='constant', cval=0)

    # 计算局部方差
    variance = mean_arr_squared - mean_arr ** 2

    # 计算局部标准差
    std_dev = np.sqrt(variance)

    # 提取窗口完全在输入矩阵范围内那些位置的结果
    arr_shape = np.array(arr.shape)
    roi_size = np.array(roi_size)
    half_roi = roi_size // 2
    valid_start = half_roi
    valid_end = arr_shape - (roi_size - 1) // 2

    # 构建有效区域的切片
    valid_slices = tuple(slice(start, end) for start, end in zip(valid_start, valid_end))

    # 提取有效位置的均值和标准差
    mean_arr_window_within_input = mean_arr[valid_slices]
    std_dev_window_within_input = std_dev[valid_slices]

    # 计算每个窗口中所有元素都大于 0的那些位置
    window_mask = roi_all_larger_than_zero(arr, roi_size=roi_size)

    # 找出均值非 NA（窗口中所有元素 > 0）且均值不为零的位置
    valid_mask = (~np.isnan(mean_arr_window_within_input)) & (mean_arr_window_within_input > 0) & window_mask

    # 计算变异系数（CV）
    cv = np.full_like(mean_arr_window_within_input, fill_value=np.nan, dtype=float)
    cv[valid_mask] = std_dev_window_within_input[valid_mask] / mean_arr_window_within_input[valid_mask]

    # 排除所有元素都是 NA 的情况
    if not np.all(np.isnan(cv)):
        min_index_linear = np.nanargmin(cv)  # 注意排除 NA 的影响
        min_index_2d = np.unravel_index(min_index_linear, cv.shape)
        best_roi_row, best_roi_col = min_index_2d

        # 获取最佳 ROI
        best_roi = arr[best_roi_row:best_roi_row + roi_h, best_roi_col:best_roi_col + roi_w]
        best_roi_mask[best_roi_row:best_roi_row + roi_h, best_roi_col:best_roi_col + roi_w] = 1

    return best_roi, best_roi_mask


def mean_if_all_positive(values):
    if np.all(values > 0):
        return np.mean(values)
    else:
        return np.nan

def mean_of_squares_if_all_positive(values):
    if np.all(values > 0):
        return np.mean(values ** 2)
    else:
        return np.nan

from scipy.ndimage import generic_filter

def find_min_cv_roi2(arr, roi_size):
    best_roi = None
    best_roi_mask = np.zeros_like(arr)

    rows, cols = arr.shape
    roi_h, roi_w = roi_size

    # 将输入数组转换为浮点数，防止整数溢出
    arr = arr.astype(float)  # 从float32转成了float64

    ########################################################
    # 计算局部均值
    mean_arr = generic_filter(arr, function=mean_if_all_positive, size=roi_size, mode='constant', cval=0)

    # 计算局部平方均值
    mean_arr_squared = generic_filter(arr, function=mean_of_squares_if_all_positive, size=roi_size, mode='constant', cval=0)


    # 计算局部方差
    variance = mean_arr_squared - mean_arr ** 2

    # 计算局部标准差
    std_dev = np.sqrt(variance)

    # 提取窗口完全在输入矩阵范围内那些位置的结果
    # Determine valid indices where the filter window is fully within the input array
    arr_shape = np.array(arr.shape)
    roi_size = np.array(roi_size)
    half_roi = roi_size // 2

    # For even-sized filters, uniform_filter behavior can be tricky due to the origin
    # We need to adjust the valid region accordingly
    valid_start = half_roi
    valid_end = arr_shape - (roi_size - 1) // 2

    # Build slices for valid region
    valid_slices = tuple(slice(start, end) for start, end in zip(valid_start, valid_end))

    # Extract the filtered values within valid positions
    mean_arr_window_within_input = mean_arr[valid_slices]
    std_dev_window_within_input = std_dev[valid_slices]


    # 找出均值非NA（窗口中所有元素>0）且均值不为零的位置
    valid_mask = (~np.isnan(mean_arr_window_within_input)) & (mean_arr_window_within_input != 0)
    # valid_mask = np.logical_and(~np.isnan(mean_arr_window_within_input), mean_arr_window_within_input != 0)

    # 计算变异系数（CV）
    # 初始化 CV 数组，形状与有效区域的均值数组相同
    cv = np.full_like(mean_arr_window_within_input, fill_value=np.nan)


    # 对均值不为零的位置计算 CV
    cv[valid_mask] = std_dev_window_within_input[valid_mask] / mean_arr_window_within_input[valid_mask]

    # 对于均值为零的位置，CV 保持为 NaN（表示 NA）


    if not np.all(np.isnan(cv)): # 排除所有元素都是NA的情况
        min_index_linear = np.nanargmin(cv) # 注意排除NA的影响
        min_index_2d = np.unravel_index(min_index_linear, cv.shape)
        best_roi_row = min_index_2d[0]
        best_roi_col = min_index_2d[1]

        #################################################
        if best_roi_row > 0 or best_roi_col > 0:
            # 获取最佳ROI
            best_roi = arr[best_roi_row:best_roi_row + roi_h, best_roi_col:best_roi_col + roi_w]

            # 构建ROI mask
            best_roi_mask[best_roi_row:best_roi_row + roi_h, best_roi_col:best_roi_col + roi_w] = 1



    return best_roi, best_roi_mask

def APHE_Washout_ZhongShan(img_2d_win, t_label_win, l_2d_win, label_index, around_dilate_size, radius, phase, data_site, rescaled, feature, A_whole_enhancement):

    Debugging = False
    if phase == 'A':
        Debugging = False
    if phase == 'V':
        Debugging = False

    auto_segmentation = True

    if feature == 'APHE':
        use_manual_parameters = False
    if feature == 'Washout':
        use_manual_parameters = True


    if auto_segmentation:
        if use_manual_parameters:
            remove_liver_dark = False
            remove_liver_dark_ratio = 12

            remove_liver_bright = False
            remove_liver_bright_ratio = 75


            around_k = 0.25
        else:
            if feature == 'APHE':
                remove_liver_dark = True
                remove_liver_dark_ratio = 12
                remove_liver_bright = True
                remove_liver_bright_ratio = 95

            if feature == 'Washout':
                remove_liver_dark = True
                remove_liver_dark_ratio = 12
                remove_liver_bright = False
                remove_liver_bright_ratio = 95

            around_k = 0.25

    else:
        remove_liver_dark = False
        remove_liver_bright = False
        around_k = 0.25


    if auto_segmentation and phase == 'A':
        best_ROI = True # 提取best_ROI

        use_best_ROI_enhancement = True # 将best_ROI用于判断动脉期是否完全强化，主要是为了区分动脉期完全强化还是渐进性强化
        whole_enhancement_intensity_k = 1.5
        whole_enhancement_area_ratio = 0.85

        use_best_ROI_washout = False
    elif auto_segmentation and phase == 'V':
        best_ROI = True  # 提取best_ROI
        use_best_ROI_enhancement = False
        enhancement_intensity_k = 1.3

        use_best_ROI_washout = True
        washout_intensity_k = 1.15
        washout_area_ratio = 0.25
    else:
        best_ROI = False  # 提取best_ROI
        use_best_ROI_enhancement = False
        use_best_ROI_washout = False


    use_best_ROI_around = False # 计算肿瘤强化强度时是否使用best_ROI作为肝背景


    remove_bright_within_tumor = False


    # if feature == 'Washout':
    #     t_label_win = array_proc.erosion(t_label_win, 0)

    # if feature == 'APHE': # liver_rim_erosion
    #     # t_label_win = array_proc.erosion(t_label_win, 3)
    #
    #     l_2d_win_erosion = array_proc.erosion(l_2d_win, 4)
    #     y, x = np.where(l_2d_win_erosion == False)[0], np.where(l_2d_win_erosion == False)[1]
    #
    #     t_label_win[y, x] = 0


    normalize = False

    t_label_win_j, t_label_win_outj = array_proc.break_label(t_label_win, label_index)  # get different label


    ####################### Get Tumor around liver ###########################
    # t_around = array_proc.get_around_ZhongShan(t_label_win_j, t_label_win_outj, around_dilate_size) # don't use liver mask
    print('APHE_testing.py', phase)
    t_around = array_proc.get_around(t_label_win_j, t_label_win_outj, l_2d_win, radius*around_k) # use liver mask

    around_pix_img_contour = copy.deepcopy(img_2d_win)
    # t_label_win_around = array_proc.dilate(t_label_win, radius*around_k)
    # y, x = np.where(l_2d_win == 0)[0], np.where(l_2d_win == 0)[1]
    # t_label_win_around[y,x] = 0

    ####################### End Get Tumor around liver ###########################




    if np.all(t_around == False):
        score = 0
        t_point_img_mean_value, inside_mean_value = 10000000, 10000000
        around_pix_img_mean_value, around_pix_img_mean_value = 10000000, 10000000
        around_mean_value = 10000000
        print('Without liver background !!')
    # else:

    # Around
    # 在肝背景上取best_ROI
    # if (feature == 'Washout') and (phase == 'V'):
    y_t, x_t = np.where(t_label_win == 1)[0], np.where(t_label_win == 1)[1]
    l_2d_win_l = copy.deepcopy(l_2d_win)
    l_2d_win_l[y_t, x_t] = 0

    y_l_t, x_l_t = np.where(l_2d_win_l == 0)[0], np.where(l_2d_win_l == 0)[1]
    dyn_2d_win_l_t = copy.deepcopy(img_2d_win)
    dyn_2d_win_l_t[y_l_t, x_l_t] = 0

    dyn_2d_win_l_t_removed_dark = copy.deepcopy(dyn_2d_win_l_t)
    dyn_2d_win_l_t_removed_bright = copy.deepcopy(dyn_2d_win_l_t)

    if Debugging:
        plt.figure()
        plt.suptitle(phase)
        plt.subplot(2, 4, 1)
        plt.imshow(img_2d_win, cmap='gray')
        plt.subplot(2, 4, 2)
        plt.imshow(dyn_2d_win_l_t, cmap='gray')
    # --------------- Remove very dark region --------------------

    if remove_liver_dark == True or remove_liver_bright == True:

        # 将二维像素值按照信号强度排序，并将信号值较小的2%置0，去暗区域
        # 将二维数组展平，并去除所有的0值
        flattened_nonzero = dyn_2d_win_l_t[dyn_2d_win_l_t != 0].flatten()

        # 如果去除0后数组为空，则直接返回原数组
        if flattened_nonzero.size == 0:
            dyn_2d_win_l_t_removed_dark = dyn_2d_win_l_t
            dyn_2d_win_l_t_removed_bright = dyn_2d_win_l_t
        else:

            if remove_liver_dark:
                # 计算需要置0的阈值，即去除0值后顶部5%的信号强度
                threshold = np.percentile(flattened_nonzero, remove_liver_dark_ratio)

                # 将所有大于或等于阈值的元素置为0
                dyn_2d_win_l_t_removed_dark[dyn_2d_win_l_t_removed_dark <= threshold] = 0

                if Debugging:
                    plt.subplot(2, 4, 3)
                    plt.imshow(dyn_2d_win_l_t_removed_dark, cmap='gray')
                    plt.title('Remove Liver Dark')

            # --------------- Remove very bright region, mainly blood vessel --------------------
            if remove_liver_bright:

                # 计算需要置0的阈值，即去除0值后顶部5%的信号强度
                threshold = np.percentile(flattened_nonzero, remove_liver_bright_ratio)

                dyn_2d_win_l_t_removed_bright[dyn_2d_win_l_t_removed_bright >= threshold] = 0

                if Debugging:
                    plt.subplot(2, 4, 4)
                    plt.imshow(dyn_2d_win_l_t_removed_bright, cmap='gray')
                    plt.title('Remove Liver Bright')

    # if feature == 'APHE' and phase == 'pre' and remove_liver_dark == True:
    #     y, x = np.where(dyn_2d_win_l_t_removed_dark == 0)[0], np.where(dyn_2d_win_l_t_removed_dark == 0)[1]
    #     dyn_2d_win_l_t[y, x] = 0


    if feature == 'Washout' and remove_liver_dark == True:
        y, x = np.where(dyn_2d_win_l_t_removed_dark == 0)[0], np.where(dyn_2d_win_l_t_removed_dark == 0)[1]
        dyn_2d_win_l_t[y, x] = 0
    # if feature == 'Washout' and remove_liver_bright == True:
    #     y, x = np.where(dyn_2d_win_l_t_removed_bright == 0)[0], np.where(dyn_2d_win_l_t_removed_bright == 0)[1]
    #     dyn_2d_win_l_t[y, x] = 0
    if feature == 'APHE' and phase == 'A' and remove_liver_bright == True:
        y, x = np.where(dyn_2d_win_l_t_removed_bright == 0)[0], np.where(dyn_2d_win_l_t_removed_bright == 0)[1]
        dyn_2d_win_l_t[y, x] = 0

    if best_ROI:
        # --------------- 在去除大血管后的肝脏区域选择ROI，用于比较待选包膜区域是否属于高亮区域 ----------------
        ROI_size = 20  # 选20*20的ROI区域
        best_roi, best_roi_mask = find_min_cv_roi(dyn_2d_win_l_t, roi_size=(ROI_size, ROI_size))
        if best_roi is None:
            best_roi, best_roi_mask = find_min_cv_roi(dyn_2d_win_l_t, roi_size=(15, 15))
            if best_roi is None:
                best_roi, best_roi_mask = find_min_cv_roi(dyn_2d_win_l_t, roi_size=(10, 10))
                if best_roi is None:
                    best_roi, best_roi_mask = find_min_cv_roi(dyn_2d_win_l_t, roi_size=(5, 5))

        dyn_2d_win_l_t_removed_best_roi_mask = display.add_contour_to_img(dyn_2d_win_l_t, best_roi_mask,
                                                                          1,
                                                                          (0, 255, 0), 1)

        if Debugging:
            plt.subplot(2, 4, 5)
            plt.imshow(dyn_2d_win_l_t_removed_best_roi_mask, cmap='gray')
            plt.title('best ROI')

            # plt.show()



    # 计算肿瘤强化强度时是使用best_ROI作为肝背景
    if use_best_ROI_around and best_ROI and best_roi is not None:  # 将每一个待选包膜区域与ROI区域进行比较
        around_pix_img = copy.deepcopy(img_2d_win)
        y, x = np.where(best_roi_mask == 0)[0], np.where(best_roi_mask == 0)[1]
        around_pix_img[y, x] = 0

    else:
        around_pix_img = copy.deepcopy(img_2d_win)
        y, x = np.where(t_around == False)[0], np.where(t_around == False)[1]
        around_pix_img[y, x] = 0



    if (feature == 'Washout') and (phase == 'V') and (remove_liver_dark == True):
        y, x = np.where(dyn_2d_win_l_t_removed_dark == 0)[0], np.where(dyn_2d_win_l_t_removed_dark == 0)[1]
        around_pix_img[y, x] = 0

    if (feature == 'Washout') and (phase == 'V') and (remove_liver_bright == True):
        y, x = np.where(dyn_2d_win_l_t_removed_bright == 0)[0], np.where(dyn_2d_win_l_t_removed_bright == 0)[1]
        around_pix_img[y, x] = 0

    if (feature == 'APHE') and (phase == 'A') and (remove_liver_bright == True):
        y, x = np.where(dyn_2d_win_l_t_removed_bright == 0)[0], np.where(dyn_2d_win_l_t_removed_bright == 0)[1]
        around_pix_img[y, x] = 0

    # if (feature == 'APHE') and (phase == 'pre') and (remove_liver_dark == True):
    #     y, x = np.where(dyn_2d_win_l_t_removed_dark == 0)[0], np.where(dyn_2d_win_l_t_removed_dark == 0)[1]
    #     around_pix_img[y, x] = 0


    ########################### Calculate mean intensity within the lesion #################
    # if Debugging:
    y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
    t_point_img = copy.deepcopy(img_2d_win)
    t_point_img[y, x] = 0  # lesion region



    exist_Tumor = (t_point_img != 0)
    t_point_img_mean_value = t_point_img.sum() / exist_Tumor.sum()  # mean of whole lesion

    # # 获取所有非零元素
    # non_zero_elements = t_point_img[exist_Tumor]
    # # 计算 75 百分位数
    # t_point_img_mean_value = np.percentile(non_zero_elements, 75)   # 75th percentile of whole lesion (用于消融实验)
    ########################### End Calculate mean intensity within the lesion #################

    # ------------------ 判断动脉期是否完全强化，主要是为了区分动脉期完全强化还是渐进性强化 ------------------------
    whole_enhancement = False
    if best_ROI and use_best_ROI_enhancement and best_roi is not None and feature == 'Washout' and phase == 'A':
        around_pix_img_ROI = copy.deepcopy(img_2d_win)
        y, x = np.where(best_roi_mask == 0)[0], np.where(best_roi_mask == 0)[1]
        around_pix_img_ROI[y, x] = 0

        exist = (around_pix_img_ROI != 0)
        ROI_mean = around_pix_img_ROI.sum() / exist.sum()  # mean of best_ROI


        t_point_img_bright_ROI = copy.deepcopy(t_point_img)
        threshold_ROI = ROI_mean*whole_enhancement_intensity_k
        t_point_img_bright_ROI[t_point_img_bright_ROI <= threshold_ROI] = 0

        exist_Tumor_Bright = (t_point_img_bright_ROI != 0)
        Bright_mean = t_point_img_bright_ROI.sum() / exist_Tumor_Bright.sum()

        enhancement_area_ratio = exist_Tumor_Bright.sum() / exist_Tumor.sum()

        if t_point_img_mean_value/ROI_mean > whole_enhancement_intensity_k or enhancement_area_ratio >= whole_enhancement_area_ratio:
            whole_enhancement = True

        if Debugging:
            if feature == 'Washout' and phase == 'A':
                plt.subplot(2, 4, 6)
                plt.imshow(t_point_img, cmap='gray')
                plt.title('Tumor')

                plt.text(5, 20, 'mean of whole lesion: ' + str(round(t_point_img_mean_value, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                plt.text(5, 30, 'mean of whole lesion/bestROI: ' + str(round(t_point_img_mean_value/ROI_mean, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                plt.subplot(2, 4, 7)
                plt.imshow(t_point_img_bright_ROI, cmap='gray')
                plt.title('Tumor Bright by 1.5*ROI Mean')

                plt.text(5, 20, 'mean of bright lesion: ' + str(round(Bright_mean, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                plt.text(5, 30, 'Enhancement Area Ratio: ' + str(round(enhancement_area_ratio, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                plt.text(5, 60,
                         'Whole Enhancement (Bright Area or Intensity): ' + str(whole_enhancement),
                         fontdict=dict(fontsize=8, color='r', family='monospace', weight='bold'))

                plt.subplot(2, 4, 8)
                plt.imshow(around_pix_img_ROI, cmap='gray')
                plt.title('around ROI')

                plt.text(5, 20, 'mean of best ROI: ' + str(round(ROI_mean, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))


    elif feature == 'Washout' and phase == 'V' and best_ROI and best_roi is not None and use_best_ROI_washout: # 如果动脉期完全强化，提取V中的暗区域来和周边肝脏比较，只要暗区域比周边肝脏信号低，即认为是有Washout；但最后返回的是OTSU_Dark_Mean
        # 提取V中暗区域
        around_pix_img_ROI = copy.deepcopy(img_2d_win)
        y, x = np.where(best_roi_mask == 0)[0], np.where(best_roi_mask == 0)[1]
        around_pix_img_ROI[y, x] = 0

        exist = (around_pix_img_ROI != 0)
        ROI_mean = around_pix_img_ROI.sum() / exist.sum()  # mean of best_ROI


        # if A_whole_enhancement: # 如果动脉期完全强化
        #     t_point_img_dark_ROI = copy.deepcopy(t_point_img)
        #     threshold_ROI = ROI_mean * washout_intensity_k
        #     t_point_img_dark_ROI[t_point_img_dark_ROI > threshold_ROI] = 0 # 提取V中的暗区域
        #
        #     exist_Tumor_Dark = (t_point_img_dark_ROI != 0)
        #     Dark_mean = t_point_img_dark_ROI.sum() / exist_Tumor_Dark.sum() # 计算V中暗区域的均值
        #
        #     if exist_Tumor_Dark.sum() / exist_Tumor.sum() < washout_area_ratio: # 如果暗区域的面积占整个肿瘤面积1/4以下，则不用V中暗区域去比，还是用整个肿瘤区域
        #         t_point_img_dark_ROI = copy.deepcopy(t_point_img)


        t_point_img_bright_ROI = copy.deepcopy(t_point_img)
        threshold_ROI = ROI_mean * enhancement_intensity_k
        t_point_img_bright_ROI[t_point_img_bright_ROI <= threshold_ROI] = 0

        exist_Tumor_Bright = (t_point_img_bright_ROI != 0)
        # Bright_mean = t_point_img_bright_ROI.sum() / exist_Tumor_Bright.sum()

        enhancement_area_ratio = exist_Tumor_Bright.sum() / exist_Tumor.sum() #计算V中亮区域占整个肿瘤的面积之比，用于和A比，判断是否渐进性强化


        if Debugging:
            if feature == 'Washout' and phase == 'V':
                plt.subplot(2, 4, 6)
                plt.imshow(t_point_img, cmap='gray')
                plt.title('Tumor')

                plt.text(5, 20, 'mean of whole lesion: ' + str(round(t_point_img_mean_value, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                plt.text(5, 30, 'mean of whole lesion/bestROI: ' + str(round(t_point_img_mean_value / ROI_mean, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                plt.subplot(2, 4, 7)
                if A_whole_enhancement:
                    print('Wholely Enhanced!!!')
                    # plt.imshow(t_point_img_dark_ROI, cmap='gray')
                    # plt.title('Whole Enhanced: Tumor Dark by 1.3*ROI Mean')
                    #
                    #
                    # plt.text(5, 20, 'mean of dark lesion: ' + str(round(Dark_mean, 2)),
                    #          fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                    # plt.text(5, 30,
                    #          'Dark mean / ROI mean: ' + str(round(Dark_mean / ROI_mean, 2)),
                    #          fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                else:
                    plt.imshow(t_point_img_bright_ROI, cmap='gray')
                    plt.title('Not Whole Enhanced: Tumor Bright by 1.5*ROI Mean')

                    # plt.text(5, 20, 'mean of bright lesion: ' + str(round(Bright_mean, 2)),
                    #          fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                    plt.text(5, 30,
                             'Enhancement Area Ratio(Not Used): ' + str(round(enhancement_area_ratio, 2)),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))


                plt.subplot(2, 4, 8)
                plt.imshow(around_pix_img_ROI, cmap='gray')
                plt.title('around ROI')

                plt.text(5, 20, 'mean of best ROI: ' + str(round(ROI_mean, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

    else:
        enhancement_area_ratio = 0

    print('A_whole_enhancement: ', A_whole_enhancement)

    if Debugging:
        plt.show()

    if (feature == 'Washout') and (phase == 'V') and (remove_bright_within_tumor == True):
        flattened_nonzero = t_point_img[t_point_img != 0].flatten()
        # 如果去除0后数组为空，则直接返回原数组
        if flattened_nonzero.size != 0:

            if Debugging:
                plt.subplot(2, 4, 7)
                plt.imshow(t_point_img, cmap='gray')

            threshold = np.percentile(flattened_nonzero, 80)
            # 将所有大于或等于阈值的元素置为0
            t_point_img[t_point_img >= threshold] = 0

            if Debugging:
                plt.subplot(2, 4, 8)
                plt.imshow(t_point_img, cmap='gray')

                plt.show()



    # otsu threshold based segmentation and get mean gray value inside tumor
    dyn_t_y, dyn_t_x = np.where(t_label_win == label_index)[0], np.where(t_label_win == label_index)[1]
    t_point = img_2d_win[dyn_t_y, dyn_t_x]  # Get foreground pixels (intensity values of each tumor pixel)
    # print('t_point: ', t_point)

    if len(t_point) > 1 and np.int0((np.max(t_point)-np.min(t_point))) > 0:
        threshold_ = array_proc.otsu(t_point)
        if threshold_ > 0:

            if normalize:
                cv2.normalize(t_point, t_point, 0, 255, cv2.NORM_MINMAX)
                t_point = t_point.astype(np.uint8)

            if not rescaled:
                ############### OTSU Threshold #######################
                threshold = array_proc.otsu(t_point)

                light_num = np.where(t_point >= threshold)[0] # positions where signal intensity is larger than the threshold
                light_pix = img_2d_win[dyn_t_y[light_num], dyn_t_x[light_num]] # signal intensities larger than the threshold
                if feature == 'APHE' and auto_segmentation and (not use_manual_parameters):
                    # tumor_OTSU_bright = np.mean(light_pix) # bright mean based on OTSU，用于消融实验
                    tumor_OTSU_bright = np.percentile(light_pix, 75)
                else:
                    tumor_OTSU_bright = np.mean(light_pix) # bright mean based on OTSU



                if normalize:
                    t_point_img_normalized = copy.deepcopy(t_point_img)
                    cv2.normalize(t_point_img, t_point_img_normalized, 0, 255, cv2.NORM_MINMAX)
                    y, x = np.where(t_point_img_normalized.astype(np.uint8) < threshold)[0], np.where(t_point_img_normalized.astype(np.uint8) < threshold)[1]
                    y2, x2 = np.where(t_point_img_normalized.astype(np.uint8) > threshold)[0], \
                           np.where(t_point_img_normalized.astype(np.uint8) > threshold)[1]
                else:
                    y, x = np.where(t_point_img < threshold)[0], np.where(t_point_img < threshold)[1]
                    y2, x2 = np.where(t_point_img > threshold)[0], np.where(t_point_img > threshold)[1]
                light_pix_img = copy.deepcopy(t_point_img)
                light_pix_img[y, x] = 0  # bright region within lesion

                dark_pix_img = copy.deepcopy(t_point_img)
                dark_pix_img[y2, x2] = 0  # dark region within lesion

                exist_tumor = (t_point_img != 0)
                exist = (dark_pix_img != 0)
                # if dark region is too small, use mena intensity of whole tumor region to represent mean intensity of dark region, to improve the reliability of the algorithm.
                if exist.sum() > 0.1*exist_tumor.sum():
                    tumor_OTSU_dark = dark_pix_img.sum() / exist.sum()
                else:
                    tumor_OTSU_dark = t_point_img_mean_value

            if Debugging:
                plt.figure()
                plt.suptitle(phase)
            if Debugging:

                plt.subplot(3, 5, 1)
                plt.imshow(img_2d_win, cmap='gray')
                # plt.imshow(result, cmap='gray')
                plt.title('Tumor ROI')

                if feature == 'Washout' and phase == 'V':
                    plt.text(5, 20, 'A Whole Enhancement: ' + str(A_whole_enhancement),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))


                plt.subplot(3, 5, 2)
                plt.imshow(t_point_img, cmap='gray')
                plt.title('lesion region')

                plt.text(5, 20, 'mean of whole lesion: ' + str(round(t_point_img_mean_value, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                if not rescaled:
                    plt.subplot(3, 5, 3)
                    plt.imshow(light_pix_img, cmap='gray')
                    # plt.imshow(result, cmap='gray')
                    plt.title('bright lesion - OTSU')

                    plt.text(5, 20, 'bright mean OTSU: ' + str(round(tumor_OTSU_bright, 2)),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                    plt.subplot(3, 5, 4)
                    plt.imshow(dark_pix_img, cmap='gray')
                    # plt.imshow(result, cmap='gray')
                    plt.title('dark lesion - OTSU')

                    plt.text(5, 20, 'dark mean OTSU: ' + str(round(tumor_OTSU_dark, 2)),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            # Adaptive Threshold
            if False:
                # Adaptive Threshold
                t_point_img_normalized = copy.deepcopy(t_point_img)
                cv2.normalize(t_point_img, t_point_img_normalized, 0, 255, cv2.NORM_MINMAX)
                kernel_size = min(adaptive_threshold_mask_tumor, 2 * np.int0(0.5 * radius) + 1)

                if kernel_size == 1:
                    kernel_size = 3

                result = cv2.adaptiveThreshold(t_point_img_normalized.astype(np.uint8), 255,
                                               cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                               kernel_size, 0)

                y, x = np.where(result == 0)[0], np.where(result == 0)[1]
                light_pix_img = copy.deepcopy(t_point_img)
                light_pix_img[y, x] = 0  # bright region within lesion
                exist = (light_pix_img != 0)
                bright_mean_adaptive_threshold = light_pix_img.sum() / exist.sum()

                if Debugging:
                    plt.subplot(3, 5, 6)
                    plt.imshow(light_pix_img, cmap='gray')
                    plt.title('bright lesion adaptive')

                    plt.text(5, 20, 'bright mean adaptive: ' + str(round(bright_mean_adaptive_threshold, 2)),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))



                # Other methods to calculate tumor inside mean
                # Method 1: Dark lesion
                y_inside, x_inside = np.where(result == 255)[0], np.where(result == 255)[1]
                region_inside = copy.deepcopy(t_point_img)
                region_inside[y_inside, x_inside] = 0

                if (np.max(region_inside) == 0):
                    inside_mean_value = 0
                else:
                    exist = (region_inside != 0)
                    inside_mean_value = region_inside.sum() / exist.sum()
                if Debugging:
                    plt.subplot(3, 5, 7)
                    plt.imshow(region_inside, cmap='gray')
                    plt.title('dark lesion adaptive')

                    plt.text(5, 20, 'dark mean adaptive: ' + str(round(inside_mean_value, 2)),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))




                ########################################## Around ##############################################################
                # Other Around Method

                # # Tumor
                # y, x = np.where(t_point_img == 0)[0], np.where(t_point_img == 0)[1]
                # t_point_img = copy.deepcopy(img_2d_win)
                # t_point_img[y, x] = 0
                #
                # exist = (t_point_img != 0)
                # t_point_img_mean_value = t_point_img.sum() / exist.sum()
                #
                # if Debugging:
                #     plt.subplot(2, 4, 6)
                #     plt.imshow(t_point_img, cmap='gray')
                #     plt.title('mean of region inside, bright')
                #
                #     plt.text(5, 50, 'mean of lesion region' + str(round(t_point_img_mean_value, 2)),
                #              fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))



            # Around

            # dyn_2d_win_liver_contour = display.add_contour_to_img(around_pix_img_contour, t_label_win, 1,
            #                                                       (102,204,153), # Green (0, 255, 0) #'#66CC99' (102,204,153)
            #                                                       1)
            y,x = np.where(t_label_win == 1)[0], np.where(t_label_win == 1)[1]
            l_2d_win_no_tumor = copy.deepcopy(l_2d_win)
            l_2d_win_no_tumor[y,x] = 0
            if Debugging:
                dyn_2d_win_liver_contour = display.add_contour_to_img(around_pix_img_contour, l_2d_win_no_tumor, 1,
                                                                      (102, 153, 153),
                                                                      # Red (255, 0, 0) # '#669999'(102, 153, 153)
                                                                      1)

                dyn_2d_win_liver_contour = display.add_contour_to_img(dyn_2d_win_liver_contour, light_pix_img, 0,
                                                                      (102, 204, 153),
                                                                      # Green (0, 255, 0) #'#66CC99' (102,204,153)
                                                                      1)



                # dyn_2d_win_liver_contour = display.add_contour_to_img(dyn_2d_win_liver_contour, t_label_win_around, 0,
                #                                                       (0, 0, 255),
                #                                                       1)

                # if Debugging:
                #     plt.subplot(2, 5, 9)
                #     plt.imshow(dyn_2d_win_liver_contour, cmap='gray')
                #     plt.title('around')

                # if Debugging:
                #     plt.subplot(2, 5, 10)
                #     plt.imshow(dyn_2d_win_liver_contour, cmap='gray')
                #     plt.title('around')
                #     # plt.show()
                # plt.figure()
                # plt.subplot(1, 3, 1)
                # plt.imshow(dyn_2d_win_liver_contour, cmap='gray')
                # plt.subplot(1, 3, 2)
                # plt.imshow(around_pix_img, cmap='gray')

                dyn_2d_win_liver_contour = display.add_contour_to_img(dyn_2d_win_liver_contour, around_pix_img, 0,
                                                                      (255, 255, 0),
                                                                      1)

                # plt.subplot(1,3,3)
                # plt.imshow(dyn_2d_win_liver_contour, cmap='gray')
                # plt.show()

            if Debugging:
                plt.subplot(3, 5, 5)
                plt.imshow(dyn_2d_win_liver_contour, cmap='gray')
                plt.title('around')



            y, x = np.where(around_pix_img == 0)[0], np.where(around_pix_img == 0)[1]
            around_pix_img = copy.deepcopy(img_2d_win)
            around_pix_img[y, x] = 0



            exist = (around_pix_img != 0)
            around_pix_img_mean_value = around_pix_img.sum() / exist.sum()

            if Debugging:
                plt.subplot(3, 5, 8)
                plt.imshow(around_pix_img, cmap='gray')
                plt.title('around')
                plt.text(5, 20, 'whole around mean: ' + str(round(around_pix_img_mean_value, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                # plt.text(5, 50, 'Method1 - lesion mean / around mean: ' + str(round(t_point_img_mean_value / around_pix_img_mean_value, 2)),
                #          fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                plt.text(5, 50, 'lesion mean / around mean: ' + str(
                    round(tumor_OTSU_bright / around_pix_img_mean_value, 2)),
                         fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            # Adaptive Threshold Segmentation for Backaround
            if False:
                # Adaptive Threshold Segmentation for Backaround
                around_pix_img_normalized = copy.deepcopy(around_pix_img)
                cv2.normalize(around_pix_img, around_pix_img_normalized, 0, 255, cv2.NORM_MINMAX)
                kernel_size = min(adaptive_threshold_mask_around, 2 * np.int0(0.5 * radius) + 1)
                if kernel_size == 1:
                    kernel_size = 3
                result_around = cv2.adaptiveThreshold(around_pix_img_normalized.astype(np.uint8), 255,
                                                      cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                                      kernel_size,
                                                      0)

                y, x = np.where(result_around == 0)[0], np.where(result_around == 0)[1]
                y2, x2 = np.where(result_around > 0)[0], np.where(result_around > 0)[1]

                around_pix_img_copy = copy.deepcopy(around_pix_img)
                around_pix_img_copy[y, x] = 0  # bright around

                around_pix_img_dark = copy.deepcopy(around_pix_img)
                around_pix_img_dark[y2, x2] = 0  # dark around



                # Method 3
                y_around, x_around = np.where(around_pix_img_copy == 0)[0], np.where(around_pix_img_copy == 0)[1]
                region_around = copy.deepcopy(img_2d_win)
                region_around[y_around, x_around] = 0 # bright around Adaptive

                exist_around_selected = (region_around != 0)
                around_mean_value = region_around.sum() / exist_around_selected.sum()

                y_around, x_around = np.where(around_pix_img_dark == 0)[0], np.where(around_pix_img_dark == 0)[1]
                region_around_dark = copy.deepcopy(img_2d_win)
                region_around_dark[y_around, x_around] = 0 # dark around Adaptive

                exist_around_selected = (region_around_dark != 0)
                around_mean_value_dark = region_around_dark.sum() / exist_around_selected.sum()


                exist_lesion_bright = (light_pix_img != 0)
                washout_around_mean_value = (light_pix_img.sum() + region_around.sum()) / (
                            exist_lesion_bright.sum() + exist_around_selected.sum())

                if Debugging:
                    plt.subplot(3, 5, 9)
                    plt.imshow(around_pix_img_copy, cmap='gray')
                    plt.title('bright around adaptive')
                    plt.text(5, 20, 'Bright Around Adaptive: ' + str(round(around_mean_value, 2)),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                    plt.text(5, 40, 'adaptive- dark lesion / bright around: ' ,
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                    plt.text(5, 50, str(round(inside_mean_value / around_mean_value, 2)),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                    plt.text(5, 70, 'washout-score3: ' ,
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                    plt.text(5, 80, str(round(inside_mean_value / washout_around_mean_value, 2)),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                    plt.subplot(3, 5, 10)
                    plt.imshow(around_pix_img_dark, cmap='gray')
                    plt.title('dark around adaptive')
                    plt.text(5, 20, 'Dark Around Adaptive: ' + str(round(around_mean_value_dark, 2)),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                # End Other Around Method

                # get mean +/- std for liver around, get regions whthin which the gray value is between mean - std and mean + std
                # and get mean gray value of liver region around tumor
                dyn_around_y, dyn_around_x = np.where(t_around == True)[0], np.where(t_around == True)[1]
                around_pix = img_2d_win[dyn_around_y, dyn_around_x]
                around_std = np.std(around_pix, ddof=1)
                around_mean = np.mean(around_pix)
                around_min = around_mean - around_std
                around_max = around_mean + around_std
                pix1 = np.where(around_pix >= around_min)[0]
                around1 = around_pix[pix1]
                pix2 = np.where(around1 <= around_max)[0]
                around2 = around1[pix2]

                around_mean_min_max = np.mean(around2)
                # around_mean_all = np.mean(around_pix)  # around_pix4， including that outside max and min

                # calculate APHE score
                # score = inside_mean / around_mean_min_max  # aPHEScore2

                if Debugging:
                    around_pix_img_copy = copy.deepcopy(around_pix_img)
                    pix1x, pix1y = np.where(around_pix_img < around_min)[0], np.where(around_pix_img < around_min)[1]
                    around_pix_img_copy[pix1x, pix1y] = 0
                    pix2x, pix2y = np.where(around_pix_img > around_max)[0], np.where(around_pix_img > around_max)[1]
                    around_pix_img_copy[pix2x, pix2y] = 0

                    plt.subplot(3, 5, 11)
                    plt.imshow(around_pix_img_copy, cmap='gray')
                    plt.title('Around: mean + - std')

                    plt.text(5, 20, 'around mean min max: ' + str(round(around_mean_min_max, 2)),
                             fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))



                # ---------------------------- Method --------------------------------------------
                # lesion_dark = t_point_img - light_pix_img

                # y_inside_bright, x_inside_bright = np.where(light_pix_img == 0)[0], np.where(light_pix_img == 0)[1]
                # region_inside_bright = copy.deepcopy(img_2d_win)
                # region_inside_bright[y_inside_bright, x_inside_bright] = 0
                #
                # exist_lesion_bright = (region_inside_bright != 0)
                # lesion_bright_mean_value = region_inside_bright.sum() / exist_lesion_bright.sum()
                # if Debugging:
                #     plt.subplot(2,5,9)
                #     plt.imshow(region_inside_bright, cmap='gray')
                #     plt.title('region inside bright')
                #     plt.show()

                # washout_around_mean_value = (region_inside_bright.sum() + region_around.sum()) / (
                            # exist_lesion_bright.sum() + exist_around_selected.sum())

                # ---------------------------- End Method --------------------------------------------


                aPHEScore1 = round(t_point_img_mean_value / around_pix_img_mean_value,2)  # whole tunor/liver around
                if not rescaled:
                    aPHEScore2 = round(tumor_OTSU_bright / around_pix_img_mean_value, 2)  # tunor bright/liver
                    aPHEScore3 = round(tumor_OTSU_bright / around_mean_min_max, 2)  # tunor bright/liver around between std-mean and std+mean

                aPHEScore4 = t_point_img_mean_value - around_pix_img_mean_value

                # washoutScore1 = round(t_point_img_mean_value / around_pix_img_mean_value, 2)  # whole tunor/liver around

                if (np.max(region_inside) > 0):
                    washoutScore2 = round(inside_mean_value / around_mean_value, 2)  # darkinsidetumor/brightoutersideliver

                    washoutScore3 = round(inside_mean_value / washout_around_mean_value, 2)  # darkinsidetumor/brighttumor+brightoutsideliver
                else:
                    washoutScore2 = 0
                    washoutScore3 = 0

                # score = aPHEScore1

            if Debugging:
                plt.show()

            if data_site == 'ZhongShan' or data_site == 'SuZhou' or data_site == 'PHC':
                # if phase == 'pre' or phase == 'A':
                #     if feature == 'APHE':
                #         # return t_point_img_mean_value, around_pix_img_mean_value # 用于消融实验
                #         return tumor_OTSU_bright, around_pix_img_mean_value
                #     if feature == 'Washout' and phase == 'pre':
                #         return t_point_img_mean_value, around_pix_img_mean_value
                #
                #     if feature == 'Washout' and phase == 'A':
                #         if auto_segmentation:
                #             return tumor_OTSU_bright, around_pix_img_mean_value, enhancement_area_ratio, whole_enhancement  # For Wash in
                #         else:
                #             return t_point_img_mean_value, around_pix_img_mean_value, enhancement_area_ratio, whole_enhancement # For Wash in


                # if phase == 'V':
                #     if auto_segmentation and A_whole_enhancement:
                #         # return t_point_img_mean_value, around_pix_img_mean_value, enhancement_area_ratio # 用于消融实验, 对于主中心，几乎没啥影响
                #         return tumor_OTSU_dark, around_pix_img_mean_value, enhancement_area_ratio
                #     else:
                #         return t_point_img_mean_value, around_pix_img_mean_value, enhancement_area_ratio
                #     # return tumor_OTSU_bright, around_pix_img_mean_value

                return tumor_OTSU_bright, around_pix_img_mean_value, t_point_img_mean_value, tumor_OTSU_dark, enhancement_area_ratio, whole_enhancement


            if data_site == 'ZheYi':
                if phase == 'pre':
                    return t_point_img_mean_value, around_pix_img_mean_value
                if phase == 'A':
                    return tumor_OTSU_bright, around_pix_img_mean_value
                if phase == 'V':
                    return tumor_OTSU_dark, around_mean_value
                if phase == 'D':
                    return tumor_OTSU_dark, around_mean_value
        else:
            if feature == 'Washout' and phase == 'A':
                return 1000, 1, 1000, 1000, enhancement_area_ratio, whole_enhancement
            elif phase == 'V':
                return 1000, 1, enhancement_area_ratio
            else:
                if feature == 'APHE':
                    return 1, 1000, 1, 1, 0, False
                else:
                    return 1000, 1, 0
    else:
        if feature == 'Washout' and phase == 'A':
            return 1000, 1, 1000, 1000, enhancement_area_ratio, whole_enhancement
        elif phase == 'V':
            return 1000, 1, enhancement_area_ratio
        else:
            if feature == 'APHE':
                return 1, 1000, 1, 1, 0, False
            else:
                return 1000, 1, 0

def Washout_ZhongShan(img_2d_win, t_label_win, l_2d_win, label_index, around_dilate_size, radius, phase):
    Debugging = False
    normalize = False
    adaptive_threshold_mask_tumor = 1 # 35
    adaptive_threshold_mask_around = 15

    # Smoothing for lesion region
    # img_2d_win = cv2.GaussianBlur(img_2d_win, (9, 9), 0)

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(img_2d_win, cmap='gray')
    # plt.title('ROI')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(img_2d_win, cmap='gray')
    # plt.title('ROI smoothed')
    #
    # plt.show()

    if Debugging:
        plt.figure()

    t_label_win_j, t_label_win_outj = array_proc.break_label(t_label_win, label_index)  # get different label

    # t_around = array_proc.get_around_ZhongShan(t_label_win_j, t_label_win_outj, around_dilate_size)
    t_around = array_proc.get_around_ZhongShan(t_label_win_j, t_label_win_outj, l_2d_win, radius*0.25)

    if np.all(t_around == False):
        score = 0
        print('Without liver background !!')
    else:
        # otsu threshold based segmentation and get mean gray value inside tumor
        dyn_t_y, dyn_t_x = np.where(t_label_win == label_index)[0], np.where(t_label_win == label_index)[1]
        t_point = img_2d_win[dyn_t_y, dyn_t_x]
        # if Debugging:
        y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
        t_point_img = copy.deepcopy(img_2d_win)
        t_point_img[y, x] = 0  # lesion region

        exist = (t_point_img != 0)
        t_point_img_mean_value = t_point_img.sum() / exist.sum() # mean of whole lesion


        if normalize:
            cv2.normalize(t_point, t_point, 0, 255, cv2.NORM_MINMAX)
            t_point = t_point.astype(np.uint8)


        threshold = array_proc.otsu(t_point)

        light_num = np.where(t_point >= threshold)[0]
        light_pix = img_2d_win[dyn_t_y[light_num], dyn_t_x[light_num]]
        inside_mean = np.mean(light_pix) # bright mean based on OTSU

        dark_num = np.where(t_point < threshold)[0]
        dark_pix = img_2d_win[dyn_t_y[dark_num], dyn_t_x[dark_num]]
        inside_mean_dark = np.mean(dark_pix) # dark mean based on OTSU


        if normalize:
            t_point_img_normalized = copy.deepcopy(t_point_img)
            cv2.normalize(t_point_img, t_point_img_normalized, 0, 255, cv2.NORM_MINMAX)
            y, x = np.where(t_point_img_normalized.astype(np.uint8) < threshold)[0], np.where(t_point_img_normalized.astype(np.uint8) < threshold)[1]
        else:
            y, x = np.where(t_point_img < threshold)[0], np.where(t_point_img < threshold)[1]
        light_pix_img = copy.deepcopy(t_point_img)
        light_pix_img[y, x] = 0  # bright region within lesion

        y, x = np.where(t_point_img >= threshold)[0], np.where(t_point_img >= threshold)[1]
        dark_pix_img = copy.deepcopy(t_point_img)
        dark_pix_img[y, x] = 0  # dark region within lesion

        if Debugging:

            plt.subplot(2, 5, 1)
            plt.imshow(img_2d_win, cmap='gray')
            # plt.imshow(result, cmap='gray')
            plt.title('Tumor ROI')


            plt.subplot(2, 5, 2)
            plt.imshow(t_point_img, cmap='gray')
            plt.title('lesion region')

            plt.text(5, 20, 'mean of whole lesion: ' + str(round(t_point_img_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))


            plt.subplot(2, 5, 3)
            plt.imshow(light_pix_img, cmap='gray')
            # plt.imshow(result, cmap='gray')
            plt.title('bright lesion - OTSU')

            plt.text(5, 20, 'bright mean OTSU: ' + str(round(inside_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            plt.subplot(2, 5, 4)
            plt.imshow(dark_pix_img, cmap='gray')
            # plt.imshow(result, cmap='gray')
            plt.title('dark lesion - OTSU')

            plt.text(5, 20, 'dark mean OTSU: ' + str(round(inside_mean_dark, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

        # Adaptive Threshold
        t_point_img_normalized = copy.deepcopy(t_point_img)
        cv2.normalize(t_point_img, t_point_img_normalized, 0, 255, cv2.NORM_MINMAX)
        kernel_size = min(adaptive_threshold_mask_tumor, 2 * np.int0(0.5 * radius) + 1)
        if kernel_size == 1:
            kernel_size = 3

        result = cv2.adaptiveThreshold(t_point_img_normalized.astype(np.uint8), 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       kernel_size, 0)

        if Debugging:
            y, x = np.where(result == 0)[0], np.where(result == 0)[1]
            light_pix_img_adaptive = copy.deepcopy(t_point_img)
            light_pix_img_adaptive[y, x] = 0  # bright region within lesion

            plt.subplot(2, 5, 5)
            plt.imshow(light_pix_img_adaptive, cmap='gray')
            plt.title('bright lesion adaptive')

            exist = (light_pix_img_adaptive != 0)
            bright_mean_adaptive = light_pix_img_adaptive.sum() / exist.sum()

            plt.text(5, 20, 'bright mean adaptive: ' + str(round(bright_mean_adaptive, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

        # Other methods to calculate tumor inside mean
        # Method 1: Dark lesion
        y_inside, x_inside = np.where(result == 255)[0], np.where(result == 255)[1]
        region_inside_dark = copy.deepcopy(t_point_img)
        region_inside_dark[y_inside, x_inside] = 0

        if (np.max(region_inside_dark) == 0):
            inside_mean_dark_adaptive = 0
        else:
            exist = (region_inside_dark != 0)
            inside_mean_dark_adaptive = region_inside_dark.sum() / exist.sum()
        if Debugging:
            plt.subplot(2, 5, 6)
            plt.imshow(region_inside_dark, cmap='gray')
            plt.title('inside dark mean adaptive')

            plt.text(5, 20, 'dark mean adaptive: ' + str(round(inside_mean_dark_adaptive, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))




        ########################################## Around ##############################################################
        # Other Around Method

        # # Tumor
        # y, x = np.where(t_point_img == 0)[0], np.where(t_point_img == 0)[1]
        # t_point_img = copy.deepcopy(img_2d_win)
        # t_point_img[y, x] = 0
        #
        # exist = (t_point_img != 0)
        # t_point_img_mean_value = t_point_img.sum() / exist.sum()
        #
        # if Debugging:
        #     plt.subplot(2, 4, 6)
        #     plt.imshow(t_point_img, cmap='gray')
        #     plt.title('mean of region inside, bright')
        #
        #     plt.text(5, 50, 'mean of lesion region' + str(round(t_point_img_mean_value, 2)),
        #              fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))



        # Around
        y, x = np.where(t_around == False)[0], np.where(t_around == False)[1]
        around_pix_img = copy.deepcopy(img_2d_win)
        around_pix_img[y, x] = 0
        y, x = np.where(around_pix_img == 0)[0], np.where(around_pix_img == 0)[1]
        around_pix_img = copy.deepcopy(img_2d_win)
        around_pix_img[y, x] = 0

        exist = (around_pix_img != 0)
        around_pix_img_mean_value = around_pix_img.sum() / exist.sum()

        if Debugging:
            plt.subplot(2, 5, 7)
            plt.imshow(around_pix_img, cmap='gray')
            plt.title('around')
            plt.text(5, 20, 'whole around mean: ' + str(round(around_pix_img_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            # plt.text(5, 50, 'Method1 - lesion mean / around mean: ' + str(round(t_point_img_mean_value / around_pix_img_mean_value, 2)),
            #          fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 50, 'lesion mean / around mean: ' + str(
                round(inside_mean / around_pix_img_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))


        # Adaptive Threshold Segmentation for Backaround
        around_pix_img_normalized = copy.deepcopy(around_pix_img)
        cv2.normalize(around_pix_img, around_pix_img_normalized, 0, 255, cv2.NORM_MINMAX)
        kernel_size = min(adaptive_threshold_mask_around, 2 * np.int0(0.5 * radius) + 1)
        if kernel_size == 1:
            kernel_size = 3
        result_around = cv2.adaptiveThreshold(around_pix_img_normalized.astype(np.uint8), 255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                              kernel_size,
                                              0)

        y, x = np.where(result_around == 0)[0], np.where(result_around == 0)[1]

        around_pix_img_copy = copy.deepcopy(around_pix_img)
        around_pix_img_copy[y, x] = 0  # bright around



        # Method 3
        y_around, x_around = np.where(around_pix_img_copy == 0)[0], np.where(around_pix_img_copy == 0)[1]
        region_around = copy.deepcopy(img_2d_win)
        region_around[y_around, x_around] = 0 # bright around

        exist_around_selected = (region_around != 0)
        around_mean_value = region_around.sum() / exist_around_selected.sum()

        exist_lesion_bright = (light_pix_img != 0)
        washout_around_mean_value = (light_pix_img.sum() + region_around.sum()) / (
                    exist_lesion_bright.sum() + exist_around_selected.sum())

        if Debugging:
            plt.subplot(2, 5, 8)
            plt.imshow(around_pix_img_copy, cmap='gray')
            plt.title('bright around adaptive')
            plt.text(5, 20, 'Washout Around: ' + str(round(around_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 40, 'adaptive- dark lesion / bright around: ' ,
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 50, str(round(inside_mean_dark_adaptive / around_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            plt.text(5, 70, 'washout-score3: ' ,
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 80, str(round(inside_mean_dark_adaptive / washout_around_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

        # End Other Around Method

        # get mean +/- std for liver around, get regions whthin which the gray value is between mean - std and mean + std
        # and get mean gray value of liver region around tumor
        dyn_around_y, dyn_around_x = np.where(t_around == True)[0], np.where(t_around == True)[1]
        around_pix = img_2d_win[dyn_around_y, dyn_around_x]
        around_std = np.std(around_pix, ddof=1)
        around_mean = np.mean(around_pix)
        around_min = around_mean - around_std
        around_max = around_mean + around_std
        pix1 = np.where(around_pix >= around_min)[0]
        around1 = around_pix[pix1]
        pix2 = np.where(around1 <= around_max)[0]
        around2 = around1[pix2]

        around_mean_min_max = np.mean(around2)
        # around_mean_all = np.mean(around_pix)  # around_pix4， including that outside max and min

        # calculate APHE score
        # score = inside_mean / around_mean_min_max  # aPHEScore2

        if Debugging:
            around_pix_img_copy = copy.deepcopy(around_pix_img)
            pix1x, pix1y = np.where(around_pix_img < around_min)[0], np.where(around_pix_img < around_min)[1]
            around_pix_img_copy[pix1x, pix1y] = 0
            pix2x, pix2y = np.where(around_pix_img > around_max)[0], np.where(around_pix_img > around_max)[1]
            around_pix_img_copy[pix2x, pix2y] = 0

            plt.subplot(2, 5, 9)
            plt.imshow(around_pix_img_copy, cmap='gray')
            plt.title('APHE around: mean + - std')

            plt.text(5, 20, 'around mean min max: ' + str(round(around_mean_min_max, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))



        # ---------------------------- Method --------------------------------------------
        # lesion_dark = t_point_img - light_pix_img

        # y_inside_bright, x_inside_bright = np.where(light_pix_img == 0)[0], np.where(light_pix_img == 0)[1]
        # region_inside_bright = copy.deepcopy(img_2d_win)
        # region_inside_bright[y_inside_bright, x_inside_bright] = 0
        #
        # exist_lesion_bright = (region_inside_bright != 0)
        # lesion_bright_mean_value = region_inside_bright.sum() / exist_lesion_bright.sum()
        # if Debugging:
        #     plt.subplot(2,5,9)
        #     plt.imshow(region_inside_bright, cmap='gray')
        #     plt.title('region inside bright')
        #     plt.show()

        # washout_around_mean_value = (region_inside_bright.sum() + region_around.sum()) / (
                    # exist_lesion_bright.sum() + exist_around_selected.sum())

        # ---------------------------- End Method --------------------------------------------


        aPHEScore1 = round(t_point_img_mean_value / around_pix_img_mean_value,2)  # whole tunor/liver around
        aPHEScore2 = round(inside_mean / around_pix_img_mean_value, 2)  # tunor bright/liver
        aPHEScore3 = round(inside_mean / around_mean_min_max, 2)  # tunor bright/liver around between std-mean and std+mean

        aPHEScore4 = t_point_img_mean_value - around_pix_img_mean_value

        # washoutScore1 = round(t_point_img_mean_value / around_pix_img_mean_value, 2)  # whole tunor/liver around

        if (np.max(region_inside_dark) > 0):
            washoutScore2 = round(inside_mean_dark_adaptive / around_mean_value, 2)  # darkinsidetumor/brightoutersideliver

            washoutScore3 = round(inside_mean_dark_adaptive / washout_around_mean_value, 2)  # darkinsidetumor/brighttumor+brightoutsideliver
        else:
            washoutScore2 = 0
            washoutScore3 = 0

    if Debugging:
        plt.show()
    score = aPHEScore1
    if phase == 'pre' or phase == 'A':
        return t_point_img_mean_value, around_pix_img_mean_value
    if phase == 'V':
        # return t_point_img_mean_value, around_pix_img_mean_value, inside_mean_dark_adaptive, around_mean_value
        return t_point_img_mean_value, around_pix_img_mean_value, inside_mean_dark, around_mean_value



def APHE_Washout(img_2d_win, t_label_win, l_2d_win, label_index, around_dilate_size, radius):
    Debugging = True
    normalize = True

    adaptive_threshold_mask_tumor = 35
    adaptive_threshold_mask_around = 15

    t_label_win_j, t_label_win_outj = array_proc.break_label(t_label_win, label_index)  # get different label

    t_around = array_proc.get_around(t_label_win_j, t_label_win_outj, l_2d_win, around_dilate_size)

    if np.all(t_around == False):
        score = 0
        print('Without liver background !!')
    else:
        # otsu threshold based segmentation and get mean gray value inside tumor
        dyn_t_y, dyn_t_x = np.where(t_label_win == label_index)[0], np.where(t_label_win == label_index)[1]
        t_point = img_2d_win[dyn_t_y, dyn_t_x]
        if Debugging:
            y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
            t_point_img = copy.deepcopy(img_2d_win)
            t_point_img[y, x] = 0  # lesion region
        if normalize:
            cv2.normalize(t_point, t_point, 0, 255, cv2.NORM_MINMAX)
            t_point = t_point.astype(np.uint8)


        threshold = array_proc.otsu(t_point)

        light_num = np.where(t_point >= threshold)[0]
        light_pix = img_2d_win[dyn_t_y[light_num], dyn_t_x[light_num]]
        inside_mean = np.mean(light_pix)
        if Debugging:
            y, x = np.where(t_point_img < threshold)[0], np.where(t_point_img < threshold)[1]
            light_pix_img = copy.deepcopy(t_point_img)
            light_pix_img[y, x] = 0  # bright region within lesion

            plt.subplot(2, 3, 1)
            plt.imshow(light_pix_img, cmap='gray')
            # plt.imshow(result, cmap='gray')
            plt.title('bright lesion region based on OTSU threshold segmentation')

        # Adaptive Threshold
        t_point_img_normalized = copy.deepcopy(t_point_img)
        cv2.normalize(t_point_img, t_point_img_normalized, 0, 255, cv2.NORM_MINMAX)
        result = cv2.adaptiveThreshold(t_point_img_normalized.astype(np.uint8), 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                       min(adaptive_threshold_mask_tumor, 2 * np.int0(0.5 * radius) + 1), 0)

        if Debugging:
            y, x = np.where(result == 255)[0], np.where(result == 255)[1]
            light_pix_img = copy.deepcopy(t_point_img)
            light_pix_img[y, x] = 0  # bright region within lesion

            plt.subplot(2, 3, 2)
            plt.imshow(light_pix_img, cmap='gray')
            # plt.imshow(result, cmap='gray')
            plt.title('bright lesion region based on adaptive threshold segmentation')

        # Other methods to calculate tumor inside mean
        # Method 1
        y_inside, x_inside = np.where(light_pix_img == 0)[0], np.where(light_pix_img == 0)[1]
        region_inside = copy.deepcopy(img_2d_win)
        region_inside[y_inside, x_inside] = 0

        if (np.max(region_inside) == 0):
            inside_mean_value = 0
        else:
            exist = (region_inside != 0)
            inside_mean_value = region_inside.sum() / exist.sum()
        if Debugging:
            plt.text(5, 20, str(round(inside_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
        # Method 2
        lesion_bright = t_point_img - light_pix_img

        y_inside_bright, x_inside_bright = np.where(lesion_bright == 0)[0], np.where(lesion_bright == 0)[1]
        region_inside_bright = copy.deepcopy(img_2d_win)
        region_inside_bright[y_inside_bright, x_inside_bright] = 0

        exist_lesion_bright = (region_inside_bright != 0)
        lesion_bright_mean_value = region_inside_bright.sum() / exist_lesion_bright.sum()

        ########################################## Around ##############################################################
        # Other Around Method

        # Tumor
        y, x = np.where(t_point_img == 0)[0], np.where(t_point_img == 0)[1]
        t_point_img = copy.deepcopy(img_2d_win)
        t_point_img[y, x] = 0

        exist = (t_point_img != 0)
        t_point_img_mean_value = t_point_img.sum() / exist.sum()

        if Debugging:
            plt.subplot(2, 3, 3)
            plt.imshow(t_point_img, cmap='gray')
            plt.title('lesion region')
            plt.text(5, 50, str(round(t_point_img_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

        # Around
        y, x = np.where(t_around == False)[0], np.where(t_around == False)[1]
        around_pix_img = copy.deepcopy(img_2d_win)
        around_pix_img[y, x] = 0
        y, x = np.where(around_pix_img == 0)[0], np.where(around_pix_img == 0)[1]
        around_pix_img = copy.deepcopy(img_2d_win)
        around_pix_img[y, x] = 0

        exist = (around_pix_img != 0)
        around_pix_img_mean_value = around_pix_img.sum() / exist.sum()

        if Debugging:
            plt.subplot(2, 3, 4)
            plt.imshow(around_pix_img, cmap='gray')
            plt.title('around')
            plt.text(5, 20, str(round(around_pix_img_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            plt.text(5, 50, str(round(t_point_img_mean_value / around_pix_img_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

        # Adaptive Threshold Segmentation for Backaround
        around_pix_img_normalized = copy.deepcopy(around_pix_img)
        cv2.normalize(around_pix_img, around_pix_img_normalized, 0, 255, cv2.NORM_MINMAX)
        result_around = cv2.adaptiveThreshold(around_pix_img_normalized.astype(np.uint8), 255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                              min(adaptive_threshold_mask_around, 2 * np.int0(0.5 * radius) + 1),
                                              0)  # min(35, 2*np.int0(0.5*radius)+1)

        y, x = np.where(result_around == 0)[0], np.where(result_around == 0)[1]

        around_pix_img[y, x] = 0  # bright region within lesion

        # End Other Around Method

        # get mean +/- std for liver around, get regions whthin which the gray value is between mean - std and mean + std
        # and get mean gray value of liver region around tumor
        dyn_around_y, dyn_around_x = np.where(t_around == True)[0], np.where(t_around == True)[1]
        around_pix = img_2d_win[dyn_around_y, dyn_around_x]
        around_std = np.std(around_pix, ddof=1)
        around_mean = np.mean(around_pix)
        around_min = around_mean - around_std
        around_max = around_mean + around_std
        pix1 = np.where(around_pix >= around_min)[0]
        around1 = around_pix[pix1]
        pix2 = np.where(around1 <= around_max)[0]
        around2 = around1[pix2]

        if Debugging:
            pix1x, pix1y = np.where(around_pix_img < around_min)[0], np.where(around_pix_img < around_min)[1]
            around_pix_img[pix1x, pix1y] = 0
            pix2x, pix2y = np.where(around_pix_img > around_max)[0], np.where(around_pix_img > around_max)[1]
            around_pix_img[pix2x, pix2y] = 0

            plt.subplot(2, 3, 5)
            plt.imshow(around_pix_img, cmap='gray')
            plt.title('around: between mean - std and mean + std')

        # Method 3
        y_around, x_around = np.where(around_pix_img == 0)[0], np.where(around_pix_img == 0)[1]
        region_around = copy.deepcopy(img_2d_win)
        region_around[y_around, x_around] = 0

        exist_around_selected = (region_around != 0)
        around_mean_value = region_around.sum() / exist_around_selected.sum()

        washout_around_mean_value = (region_inside_bright.sum() + region_around.sum()) / (
                    exist_lesion_bright.sum() + exist_around_selected.sum())

        if Debugging:
            plt.subplot(2,3,6)
            plt.imshow(around_pix_img, cmap='gray')
            plt.title('around_selected')

        around_mean_min_max = np.mean(around2)
        around_mean_all = np.mean(around_pix)  # around_pix4， including that outside max and min

        # calculate APHE score
        score = inside_mean / around_mean_min_max

        if Debugging:
            plt.text(5, 20, str(round(around_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            plt.text(5, 50, str(round(inside_mean_value / around_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            plt.text(5, 80, str(round(inside_mean_value / washout_around_mean_value, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

        aPHEScore1 = round(t_point_img_mean_value / around_pix_img_mean_value,2)  # whole tunor/liver around
        aPHEScore2 = round(inside_mean / around_mean_min_max, 2)  # tunor bright/liver
        aPHEScore3 = round(inside_mean / around_mean_all, 2)  # tunor bright/liver around between std-mean and std+mean

        washoutScore1 = round(t_point_img_mean_value / around_pix_img_mean_value, 2)  # whole tunor/liver around

        if (np.max(region_inside) > 0):
            washoutScore2 = round(inside_mean_value / around_mean_value, 2)  # darkinsidetumor/brightoutersideliver

            washoutScore3 = round(inside_mean_value / washout_around_mean_value, 2)  # darkinsidetumor/brighttumor+brightoutsideliver


    return score


def APHE_score_ZheYi(img_2d_win, t_label_win, l_2d_win, label_index, around_dilate_size):
    t_label_win_j, t_label_win_outj = array_proc.break_label(t_label_win, label_index)  # get different label

    t_around = array_proc.get_around(t_label_win_j, t_label_win_outj, l_2d_win, around_dilate_size)

    if np.all(t_around == False):
        score = 0
        print('Without liver background !!')
    else:
        # otsu threshold based segmentation and get mean gray value inside tumor
        dyn_t_y, dyn_t_x = np.where(t_label_win == label_index)[0], np.where(t_label_win == label_index)[1]
        t_point = img_2d_win[dyn_t_y, dyn_t_x]

        threshold = array_proc.otsu(t_point)

        light_num = np.where(t_point >= threshold)[0]
        light_pix = img_2d_win[dyn_t_y[light_num], dyn_t_x[light_num]]
        inside_mean = np.mean(light_pix)


        # get mean +/- std for liver around, get regions whthin which the gray value is between mean - std and mean + std
        # and get mean gray value of liver region around tumor
        dyn_around_y, dyn_around_x = np.where(t_around == True)[0], np.where(t_around == True)[1]
        around_pix = img_2d_win[dyn_around_y, dyn_around_x]
        around_std = np.std(around_pix, ddof=1)
        around_mean = np.mean(around_pix)
        around_min = around_mean - around_std
        around_max = around_mean + around_std
        pix1 = np.where(around_pix >= around_min)[0]
        around1 = around_pix[pix1]
        pix2 = np.where(around1 <= around_max)[0]
        around2 = around1[pix2]


        around_mean_min_max = np.mean(around2)


        # calculate APHE score
        score = inside_mean / around_mean_min_max



    return score

def APHE_score_ZhongShan(img_2d_win, t_label_win, label_index, around_dilate_size):
    t_label_win_j, t_label_win_outj = array_proc.break_label(t_label_win, label_index)  # get different label

    t_around = array_proc.get_around_ZhongShan(t_label_win_j, t_label_win_outj, around_dilate_size)

    if np.all(t_around == False):
        score = 0
        print('Without liver background !!')
    else:
        # otsu threshold based segmentation and get mean gray value inside tumor
        dyn_t_y, dyn_t_x = np.where(t_label_win == label_index)[0], np.where(t_label_win == label_index)[1]
        t_point = img_2d_win[dyn_t_y, dyn_t_x]
        threshold = array_proc.otsu(t_point)
        light_num = np.where(t_point >= threshold)[0]
        light_pix = img_2d_win[dyn_t_y[light_num], dyn_t_x[light_num]]
        inside_mean = np.mean(light_pix)

        # get mean +/- std for liver around, get regions whthin which the gray value is between mean - std and mean + std
        # and get mean gray value of liver region around tumor
        dyn_around_y, dyn_around_x = np.where(t_around == True)[0], np.where(t_around == True)[1]
        around_pix = img_2d_win[dyn_around_y, dyn_around_x]
        around_std = np.std(around_pix, ddof=1)
        around_mean = np.mean(around_pix)
        around_min = around_mean - around_std
        around_max = around_mean + around_std
        pix1 = np.where(around_pix >= around_min)[0]
        around1 = around_pix[pix1]
        pix2 = np.where(around1 <= around_max)[0]
        around2 = around1[pix2]
        around_mean_f = np.mean(around2)

        # calculate APHE score
        score = inside_mean / around_mean_f

    return score