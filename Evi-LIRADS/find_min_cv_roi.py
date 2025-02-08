import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage import generic_filter

def uniform_filter_window_within_input(arr, roi_size):
    """
    Applies a uniform filter to the input array and returns the filtered array,
    along with the subset of the filtered array where the filter window is entirely
    within the input array.

    Parameters:
    - arr: Input numpy array.
    - roi_size: Tuple specifying the size of the uniform filter window.

    Returns:
    - filtered_array: The result of applying uniform_filter to the entire array.
    - filtered_array_window_within_input: Subset of filtered_array where the filter window
      is entirely within the input array.
    """
    best_roi = None
    best_roi_mask = np.zeros_like(arr)

    rows, cols = arr.shape
    roi_h, roi_w = roi_size

    # 将输入数组转换为浮点数，防止整数溢出
    arr = arr.astype(float)  # 从float32转成了float64

    ########################################################

    # 计算局部均值
    mean_arr = uniform_filter(arr, size=roi_size, mode='constant', cval=0)

    # 计算局部平方均值
    mean_arr_squared = uniform_filter(arr ** 2, size=roi_size, mode='constant', cval=0)

    # 计算局部方差
    variance = mean_arr_squared - mean_arr ** 2

    # 计算局部标准差
    std_dev = np.sqrt(variance)

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

    # 计算变异系数（CV）
    # 初始化 CV 数组，形状与有效区域的均值数组相同
    cv = np.full_like(mean_arr_window_within_input, fill_value=np.nan)

    # 找出均值不为零的位置
    nonzero_mean_mask = mean_arr_window_within_input != 0

    # 对均值不为零的位置计算 CV
    cv[nonzero_mean_mask] = std_dev_window_within_input[nonzero_mean_mask] / mean_arr_window_within_input[
        nonzero_mean_mask]

    # 对于均值为零的位置，CV 保持为 NaN（表示 NA）

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

    print('new method:',  best_roi_row, best_roi_col)

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
def generic_filter_window_within_input(arr, roi_size):

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
    valid_mask = ~np.isnan(mean_arr_window_within_input) and mean_arr_window_within_input != 0

    # 计算变异系数（CV）
    # 初始化 CV 数组，形状与有效区域的均值数组相同
    cv = np.full_like(mean_arr_window_within_input, fill_value=np.nan)


    # 对均值不为零的位置计算 CV
    cv[valid_mask] = std_dev_window_within_input[valid_mask] / mean_arr_window_within_input[valid_mask]

    # 对于均值为零的位置，CV 保持为 NaN（表示 NA）

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

    print('new method:',  best_roi_row, best_roi_col)

    return best_roi, best_roi_mask

def find_min_cv_roi(arr, roi_size=(20, 20)):
    # 初始化变异系数的最小值为无穷大，用于比较
    min_cv = np.inf
    # 初始化最佳ROI区域为None
    best_roi = None
    best_roi_row = 0
    best_roi_col = 0
    best_roi_mask = np.zeros((arr.shape[0], arr.shape[1]))

    # 获取数组的维度
    rows, cols = arr.shape
    Mean = []
    STD = []
    CV = []
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

                Mean.append(mean)
                STD.append(std_dev)


                # 防止平均值为0的情况
                if mean > 0:
                    cv = std_dev / mean
                    CV.append(cv)
                    # 检查是否为最小CV
                    if cv < min_cv:
                        min_cv = cv
                        best_roi = current_roi
                        best_roi_row = i
                        best_roi_col = j
    # print(Mean)
    # print(STD)
    print(CV)
    print('old method: ', best_roi_row, best_roi_col)

    best_roi_mask[best_roi_row:best_roi_row + roi_size[0], best_roi_col:best_roi_col + roi_size[1]] = 1

    return best_roi, best_roi_mask


# Test the function with the provided array and roi_size
arr = np.array([
    [2, 7, 3, 3, 4, 8],
    [2, 4, 3, 5, 6, 6],
    [3, 1, 2, 2, 9, 9],
    [5, 4, 3, 4, 2, 1],
    [8, 2, 6, 8, 1, 5],
    [4, 8, 2, 8, 7, 2]
], dtype=float)

roi_size = (4, 4)

uniform_filter_window_within_input(arr, roi_size)


find_min_cv_roi(arr, roi_size=roi_size)



