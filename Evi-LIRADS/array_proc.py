import copy

import numpy as np
from skimage import morphology,measure,filters
from scipy import ndimage



import os
import file_read
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2

# 去除肿瘤标注中标签不是1的部分
def remove_t_1(t_array):
    t_array_cop=np.copy(t_array)
    tz,ty,tx=np.where(t_array>1)
    t_array_cop[tz,ty,tx]=0
    # print('position of layer > 1', 'tz: ', tz)
    # for i in range(0, tz.size()):
    #     print(t_array[tz[i]], t_array[ty[i]], t_array[tx[i]])
    return t_array_cop

# 去除肿瘤标注中较小的区域
def remove_small_t(t_array_2d,size):
    t_array_2d_cop=np.copy(t_array_2d)
    t_array_2d_cop=t_array_2d_cop.astype(bool)
    t_array_2d_max=morphology.remove_small_objects(t_array_2d_cop,min_size=size,connectivity=1) # connectivity: 邻接模式，1表示4邻接，2表示8邻接
    t_array_2d_max=t_array_2d_max.astype(t_array_2d.dtype)
    return t_array_2d_max

def get_capsule_region_skeleton(capsule_region):
    from skimage import morphology
    capsule_region_skeleton = morphology.skeletonize(capsule_region)

    return capsule_region_skeleton

def tumor_rim_length(t_label_win):
    dyn_2d_win_skeleton = copy.deepcopy(t_label_win)

    # y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
    # dyn_2d_win_skeleton[y, x] = 0

    t_label_win_erosion_1 = erosion(t_label_win, 1)
    y, x = np.where(t_label_win_erosion_1 == 1)[0], np.where(t_label_win_erosion_1 == 1)[1]

    dyn_2d_win_skeleton[y, x] = 0

    # y, x = np.where(dyn_2d_win_skeleton > 0)[0], np.where(dyn_2d_win_skeleton > 0)[1]
    #
    # dyn_2d_win_skeleton[y, x] = 1

    # true_capsule_region = display.add_contour_to_img(true_capsule_region,
    #                                                  dyn_2d_win_skeleton, 0,
    #                                                  (0, 255, 0), 1)

    exist = (dyn_2d_win_skeleton != 0)
    lesion_rim_sum = exist.sum()

    return dyn_2d_win_skeleton, lesion_rim_sum

def remove_small_length(t_array_2d,length):
    import cv2
    t_array_2d_lesions = np.uint8(t_array_2d)

    ret, binary = cv2.threshold(t_array_2d_lesions, 0, 1, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary,
                                                    cv2.RETR_LIST,
                                                    cv2.CHAIN_APPROX_SIMPLE)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    t_array_2d_large = copy.deepcopy(t_array_2d)
    for i in range(1, num_labels):
        t_array_2d_copy = copy.deepcopy(t_array_2d)
        y,x = np.where(labels != i)[0], np.where(labels != i)[1]
        t_array_2d_copy[y,x] = 0

        region_skeleton = get_capsule_region_skeleton(t_array_2d_copy)

        exist = (region_skeleton != 0)
        region_length = exist.sum()
        if region_length < length:
            y, x = np.where(labels == i)[0], np.where(labels == i)[1]
            t_array_2d_large[y,x] = 0


    return t_array_2d_large


def capsule_region_length(capsule_region):
    region_skeleton = get_capsule_region_skeleton(capsule_region)

    exist = (region_skeleton != 0)
    region_length = exist.sum()

    return region_length, region_skeleton

# 去除原肝脏图像中没有标注的层面
def get_layers_0(t_array,dyn_array,l_array):
    tz_layers=np.unique(np.where(t_array==1)[0])
    dyn_layers=dyn_array[tz_layers,:,:].astype(np.int16)
    t_layers=t_array[tz_layers,:,:]
    l_layers=l_array[tz_layers,:,:]
    return t_layers,dyn_layers,l_layers, tz_layers

def get_layers(t_array,dyn_array,l_array):
    tz_layers=np.unique(np.where(t_array==1)[0])

    dyn_layers = dyn_array[tz_layers, :, :]

    # if phase is 'APHE':
    #     dyn_layers=dyn_array[tz_layers,:,:].astype(np.short)
    # else:
    #     dyn_layers = dyn_array[tz_layers, :, :].astype(np.double)

    t_layers=t_array[tz_layers,:,:]
    l_layers=l_array[tz_layers,:,:]
    return t_layers,dyn_layers,l_layers, tz_layers



def get_layers_AP_DP(t_array,dyn2_array,dyn4_array,l_array):
    tz_layers=np.unique(np.where(t_array==1)[0])

    dyn2_layers = dyn2_array[tz_layers, :, :]
    dyn4_layers = dyn4_array[tz_layers, :, :]

    t_layers=t_array[tz_layers,:,:]
    l_layers=l_array[tz_layers,:,:]
    return t_layers,dyn2_layers,dyn4_layers,l_layers, tz_layers

def get_layers_ZheYi(t_array,l_array,pre_array,A_array,V_array, D_array):
    tz_layers=np.unique(np.where(t_array==1)[0])
    pre_layers = pre_array[tz_layers, :, :]
    A_layers = A_array[tz_layers, :, :]
    V_layers = V_array[tz_layers, :, :]
    D_layers = D_array[tz_layers, :, :]

    t_layers=t_array[tz_layers,:,:]
    l_layers=l_array[tz_layers,:,:]
    return tz_layers, pre_layers,A_layers, V_layers, D_layers, t_layers,l_layers

def get_layers_single_phase(img_array, t_array, l_array):
    tz_layers = np.unique(np.where(t_array == 1)[0])
    img_layers = img_array[tz_layers, :, :]
    t_layers = t_array[tz_layers, :, :]
    l_layers = l_array[tz_layers,:,:]

    return img_layers, t_layers, l_layers, tz_layers





# 对肿瘤标注进行膨胀

def dilate_morphologyKernel_cv2Dilation(t_array_2d,size):
    size = min(size*2+1, round(0.5*t_array_2d.shape[0])-1, round(0.5*t_array_2d.shape[1])-1)
    kernel = morphology.disk(size)
    # t_dilation = morphology.dilation(t_array_2d, kernel)

    t_array_2d_ = t_array_2d.astype(np.uint8)
    t_dilation = cv2.dilate(t_array_2d_, kernel)

    return t_dilation


def dilate(t_array_2d,size):
    size = min(size*2+1, round(0.5*t_array_2d.shape[0])-1, round(0.5*t_array_2d.shape[1])-1)
    # kernel = morphology.disk(size)
    # t_dilation = morphology.dilation(t_array_2d, kernel)

    # t_array_2d_ = t_array_2d.astype(np.uint8)
    # t_dilation = cv2.dilate(t_array_2d_, kernel)



    # t_array_2d_ = np.clip(t_array_2d, 0, 255).astype(np.uint8)

    # t_array_2d_ = t_array_2d.astype(np.uint8)
    t_array_2d_ = (t_array_2d * 255).astype(np.uint8)

    # kernel = morphology.disk(size).astype(np.uint8)
    size = round(size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * size + 1, 2 * size + 1))


    t_dilation = cv2.dilate(
        t_array_2d_,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_REPLICATE,
        borderValue=0
    )



    t_dilation = (t_dilation / 255).astype(t_array_2d.dtype)

    return t_dilation

def dilate_size(t_array_2d,size):

    size = min(size, round(0.5*t_array_2d.shape[0])-1, round(0.5*t_array_2d.shape[1])-1)
    kernel = morphology.disk(size)


    t_array_2d_ = t_array_2d.astype(np.uint8)
    t_dilation = cv2.dilate(t_array_2d_, kernel)

    return t_dilation

# 对肿瘤标注进行腐蚀
def erosion(t_array_2d,size):

    kernel = morphology.disk(size)
    # t_erosion = morphology.erosion(t_array_2d, kernel)

    # diameter = 2 * size + 1
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    #
    #
    t_array_2d_ = t_array_2d.astype(np.uint8)
    t_erosion = cv2.erode(t_array_2d_, kernel)

    return t_erosion


# 计算肿瘤标注的面积
def label(t_array_2d):
    label_t=measure.label(t_array_2d,connectivity = 1) # get connected region
    # connectivity : Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor. Accepted values are ranging from 1 to input.ndim.
    # If None, a full connectivity of input.ndim is used. [int, optional]。
    # 如果input是一个二维的图片，那么connectivity的值范围选择{1,2}，如果是None则默认是取最高的值，对于二维来说，当connectivity=1时代表4连通，当connectivity=2时代表8连通.

    # Returns:
    # labels : 和input形状一样，但是数值是标记号，所以这是一个已经标记的图片
    # num : 标记的种类数，如果输出0则只有背景，如果输出2则有两个种类或者说是连通域
    return label_t

# 裁剪出原图的肿瘤部分图像，标注图像的肿瘤部分图像
def crop0(labels,index,dyn_array_2d,l_array_2d,size):
    label_y,label_x=np.where(labels==index)[0],np.where(labels==index)[1]
    y_min=label_y.min()
    y_max=label_y.max()
    x_min=label_x.min()
    x_max=label_x.max()
    dyn_array_2d1=np.copy(dyn_array_2d)
    l_array_2d1=np.copy(l_array_2d)
    labels1=np.copy(labels)
    dyn_2d=dyn_array_2d1[y_min-size:y_max+size,x_min-size:x_max+size]
    l_2d=l_array_2d1[y_min-size:y_max+size,x_min-size:x_max+size]
    t_label_2d=labels1[y_min-size:y_max+size,x_min-size:x_max+size]
    return dyn_2d,t_label_2d,l_2d

def crop(labels,index,dyn_array_2d,l_array_2d,l_distance_2d, size):
    label_y,label_x=np.where(labels==index)[0],np.where(labels==index)[1]
    y_min=label_y.min()
    y_max=label_y.max()
    x_min=label_x.min()
    x_max=label_x.max()
    dyn_array_2d1=np.copy(dyn_array_2d)
    l_array_2d1=np.copy(l_array_2d)
    labels1=np.copy(labels)

    l_distance_2d1 = np.copy(l_distance_2d)

    dyn_2d=dyn_array_2d1[y_min-size:y_max+size,x_min-size:x_max+size]
    l_2d=l_array_2d1[y_min-size:y_max+size,x_min-size:x_max+size]
    t_label_2d=labels1[y_min-size:y_max+size,x_min-size:x_max+size]

    l_dis_2d = l_distance_2d1[y_min - size:y_max + size, x_min - size:x_max + size]
    return dyn_2d,t_label_2d,l_2d, l_dis_2d

def crop_ZheYi(labels,index,pre_array_2d,A_array_2d,V_array_2d, D_array_2d, l_array_2d, l_distance_2d, size):
    label_y,label_x=np.where(labels==index)[0],np.where(labels==index)[1]
    y_min=label_y.min()
    y_max=label_y.max()
    x_min=label_x.min()
    x_max=label_x.max()

    pre_array_2d1=np.copy(pre_array_2d)
    A_array_2d1 = np.copy(A_array_2d)
    V_array_2d1 = np.copy(V_array_2d)
    D_array_2d1 = np.copy(D_array_2d)

    l_array_2d1=np.copy(l_array_2d)
    l_distance_2d1 = np.copy(l_distance_2d)
    labels1=np.copy(labels)

    pre_2d=pre_array_2d1[y_min-size:y_max+size,x_min-size:x_max+size]
    A_2d = A_array_2d1[y_min - size:y_max + size, x_min - size:x_max + size]
    V_2d = V_array_2d1[y_min - size:y_max + size, x_min - size:x_max + size]
    D_2d = D_array_2d1[y_min - size:y_max + size, x_min - size:x_max + size]

    l_2d=l_array_2d1[y_min-size:y_max+size,x_min-size:x_max+size]
    t_label_2d=labels1[y_min-size:y_max+size,x_min-size:x_max+size]

    l_dis_2d = l_distance_2d1[y_min - size:y_max + size, x_min - size:x_max + size]
    return pre_2d,A_2d, V_2d, D_2d, t_label_2d, l_2d, l_dis_2d



# import display

def crop_single_phase(labels, index, img_array, l_array, size):

    label_y, label_x = np.where(labels == index)[0], np.where(labels == index)[1]

    y_min = label_y.min()
    y_max = label_y.max()
    x_min = label_x.min()
    x_max = label_x.max()

    # print(max(0,y_min - size), min(img_array.shape[0], y_max + size), max(0,x_min - size), min(img_array.shape[1],x_max + size))

    img_array_2d1 = np.copy(img_array)
    img_2d = img_array_2d1[max(0,y_min - size):min(img_array_2d1.shape[0], y_max + size), max(0,x_min - size):min(img_array_2d1.shape[1],x_max + size)]

    labels1 = np.copy(labels)
    t_label_2d = labels1[max(0,y_min - size):min(img_array_2d1.shape[0], y_max + size), max(0,x_min - size):min(img_array_2d1.shape[1],x_max + size)]

    l_array1 = np.copy(l_array)
    l_array_2d = l_array1[max(0,y_min - size):min(img_array_2d1.shape[0], y_max + size), max(0,x_min - size):min(img_array_2d1.shape[1],x_max + size)]



    return img_2d, t_label_2d, l_array_2d


def crop_single_phase_patch(labels, index, img_array, l_array, patch_size):

    label_y, label_x = np.where(labels == index)[0], np.where(labels == index)[1]

    y_min = np.int(label_y.min())
    y_max = np.int(label_y.max())
    x_min = np.int(label_x.min())
    x_max = np.int(label_x.max())

    size_y = (patch_size[1]-(y_max-y_min))/2
    if np.int(size_y) == size_y:
        size_y2 = size_y
    else:
        size_y = np.int(size_y)
        size_y2 = size_y + 1

    size_x = (patch_size[2]-(x_max-x_min))/2
    if np.int(size_x) == size_x:
        size_x2 = size_x
    else:
        size_x = np.int(size_x)
        size_x2 = size_x + 1

    img_array_2d1 = np.copy(img_array)
    img_2d = img_array_2d1[max(0,y_min - np.int(size_y)):min(img_array_2d1.shape[0], y_max + np.int(size_y2)), max(0,x_min - np.int(size_x)):min(img_array_2d1.shape[1],x_max + np.int(size_x2))]

    labels1 = np.copy(labels)
    t_label_2d = labels1[max(0,y_min - np.int(size_y)):min(img_array_2d1.shape[0], y_max + np.int(size_y2)), max(0,x_min - np.int(size_x)):min(img_array_2d1.shape[1],x_max + np.int(size_x2))]

    l_array1 = np.copy(l_array)
    l_array_2d = l_array1[max(0,y_min - np.int(size_y)):min(img_array_2d1.shape[0], y_max + np.int(size_y2)), max(0,x_min - np.int(size_x)):min(img_array_2d1.shape[1],x_max + np.int(size_x2))]



    return img_2d, t_label_2d, l_array_2d

def crop_single_phase_single_img(labels, index, l_array, size):
    label_y, label_x = np.where(labels == index)[0], np.where(labels == index)[1]

    y_min = label_y.min()
    y_max = label_y.max()
    x_min = label_x.min()
    x_max = label_x.max()

    l_array1 = np.copy(l_array)
    l_array_2d = l_array1[y_min - size:y_max + size, x_min - size:x_max + size]

    return l_array_2d

def crop_ZhongShan(labels_pre,labels_A, labels_V, labels_D, index,pre_array_2d,A_array_2d, V_array_2d,D_array_2d, pre_l_array_2d,A_l_array_2d, V_l_array_2d,D_l_array_2d, size):

    pre_2d_win, pre_t_label_win, pre_l_array_win = crop_single_phase(labels_pre, index, pre_array_2d,pre_l_array_2d, size)
    A_2d_win, A_t_label_win, A_l_array_win = crop_single_phase(labels_A, index, A_array_2d, A_l_array_2d, size)
    V_2d_win, V_t_label_win, V_l_array_win = crop_single_phase(labels_V, index, V_array_2d, V_l_array_2d, size)
    D_2d_win, D_t_label_win, D_l_array_win = crop_single_phase(labels_D, index, D_array_2d, D_l_array_2d, size)

    # l_array_2d1=np.copy(l_array_2d)
    # l_2d=l_array_2d1[y_min-size:y_max+size,x_min-size:x_max+size]
    # l_distance_2d1 = np.copy(l_distance_2d)
    # l_dis_2d = l_distance_2d1[y_min - size:y_max + size, x_min - size:x_max + size]

    return pre_2d_win, A_2d_win, V_2d_win, D_2d_win, pre_t_label_win, A_t_label_win, V_t_label_win, D_t_label_win, pre_l_array_win, A_l_array_win, V_l_array_win, D_l_array_win

def crop_single_phase_ZhongShan(labels_pre,index,pre_array_2d, pre_l_array_2d, size):

    pre_2d_win, pre_t_label_win, pre_l_array_win = crop_single_phase(labels_pre, index, pre_array_2d,pre_l_array_2d, size)


    # l_array_2d1=np.copy(l_array_2d)
    # l_2d=l_array_2d1[y_min-size:y_max+size,x_min-size:x_max+size]
    # l_distance_2d1 = np.copy(l_distance_2d)
    # l_dis_2d = l_distance_2d1[y_min - size:y_max + size, x_min - size:x_max + size]

    return pre_2d_win, pre_t_label_win, pre_l_array_win


def crop_AP_DP(labels,index,dyn2_array_2d,dyn4_array_2d,l_array_2d,l_distance_2d, size):
    label_y,label_x=np.where(labels==index)[0],np.where(labels==index)[1]
    y_min=label_y.min()
    y_max=label_y.max()
    x_min=label_x.min()
    x_max=label_x.max()
    dyn2_array_2d1=np.copy(dyn2_array_2d)
    dyn4_array_2d1 = np.copy(dyn4_array_2d)
    l_array_2d1=np.copy(l_array_2d)
    labels1=np.copy(labels)

    l_distance_2d1 = np.copy(l_distance_2d)

    dyn2_2d=dyn2_array_2d1[y_min-size:y_max+size,x_min-size:x_max+size]
    dyn4_2d = dyn4_array_2d1[y_min - size:y_max + size, x_min - size:x_max + size]

    l_2d=l_array_2d1[y_min-size:y_max+size,x_min-size:x_max+size]
    t_label_2d=labels1[y_min-size:y_max+size,x_min-size:x_max+size]

    l_dis_2d = l_distance_2d1[y_min - size:y_max + size, x_min - size:x_max + size]
    return dyn2_2d,dyn4_2d,t_label_2d,l_2d, l_dis_2d

# def liver(dyn_array, l_array):
#     for i in range(0, dyn_array.shape[0]):
#         l_array_2d = l_array[i,:,:]
#         dyn_array_2d = dyn_array[i,:,:]
#         y, x = np.where(l_array_2d==0)[0], np.where(l_array_2d==0)[1]
#         dyn_array_liver_2d = np.copy(dyn_array_2d)
#         dyn_array_liver_2d[y,x] = 0

def liver_3D(dyn_array, l_array):
        z, y, x = np.where(l_array==0)[0], np.where(l_array==0)[1], np.where(l_array==0)[2]
        dyn_array_liver = np.copy(dyn_array)
        dyn_array_liver[z,y,x] = 0

        return dyn_array_liver


# 填补肿瘤标注图像的空洞部分
def hole_fill(t_array_2d):
    t_array_2d_aft=ndimage.binary_fill_holes(t_array_2d)
    return t_array_2d_aft

def hole_fill_liver_mask(t_array_2d):
    t_array_2d = dilate_size(t_array_2d, 10)
    t_array_2d = erosion(t_array_2d, 10)
    t_array_2d_aft=ndimage.binary_fill_holes(t_array_2d)

    return t_array_2d_aft

# 得到不同值的肿瘤标签
def break_label(t_label_win,j):
    not_index_y,not_index_x=np.where(t_label_win!=j)[0],np.where(t_label_win!=j)[1]

    t_label_win_j=np.copy(t_label_win)

    t_label_win_j[not_index_y,not_index_x]=0

    index_y,index_x=np.where(t_label_win==j)[0],np.where(t_label_win==j)[1]

    t_label_win_outj=np.copy(t_label_win)

    t_label_win_outj[index_y,index_x]=0 ##???

    return t_label_win_j,t_label_win_outj

# 得到肿瘤标签的边缘区域
def get_around(t_label_win_j,t_label_win_outj,l_2d_win,size):
    t_label_win_j_dilate=dilate(t_label_win_j,size)
    t_label_win_j_around=np.logical_xor(t_label_win_j_dilate,t_label_win_j) # 异或算符的值为真仅当两个运算元中恰有一个的值为真，而另外一个的值为非真 (Dilate后和之前比多出来的部分)
    x=np.logical_and(t_label_win_j_around,l_2d_win)
    y=np.logical_not(t_label_win_outj)
    result=np.logical_and(x,y)

    return result

def get_around_ZhongShan(t_label_win_j,t_label_win_outj,l_2d_win,size):
    t_label_win_j_dilate=dilate(t_label_win_j,size)
    t_label_win_j_around=np.logical_xor(t_label_win_j_dilate,t_label_win_j)
    # x=np.logical_and(t_label_win_j_around,l_2d_win)
    y=np.logical_not(t_label_win_outj)
    result=np.logical_and(t_label_win_j_around,y)
    return result

def get_around_liver_mask(t_label_win_j,t_label_win_outj,l_2d_win,size):
    t_label_win_j_dilate=dilate(t_label_win_j,size)
    t_label_win_j_around=np.logical_xor(t_label_win_j_dilate,t_label_win_j)
    x=np.logical_and(t_label_win_j_around,l_2d_win)
    y=np.logical_not(t_label_win_outj)
    result=np.logical_and(x,y)
    return result

#进行otsu分割 
def otsu(dyn_2d_t):
    hist, bin_edges = np.histogram(dyn_2d_t, bins=np.int0((dyn_2d_t.max()-dyn_2d_t.min())))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    hist = hist.astype(float)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    if len(variance12) > 0:
        idx = np.argmax(variance12)
        threshold = bin_centers[:-1][idx]
        return threshold
    else:
        return 0


def region_selection(regions):
    import cv2
    t_array_2d_lesions = np.uint8(regions)

    ret, binary = cv2.threshold(t_array_2d_lesions, 0, 1, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary,
                                                    cv2.RETR_LIST,
                                                    cv2.CHAIN_APPROX_SIMPLE)

    #
    # center, radius = cv2.minEnclosingCircle(contours[0])
    # center = np.int0(center)

    # print('i is: ', i)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)


    # Lesion_details = []
    pixels_num = []

    for l in range(1, num_labels):

        # lesion_details = [round(centroids[l][0], 2),
        #                   round(centroids[l][1], 2), round(stats[l][4], 2)]
        # Lesion_details.append(lesion_details)

        pixels_num.append(round(stats[l][4], 2)) # stats会包含5个参数分别为x,y,h,w,s, s是labels对应的连通区域的像素个数

        # print('area is: ', round(stats[l + 1][4], 2))
        # y = np.where(labels == l+1)[0]
        # print('total pixels are: ', len(y))


    # sort based on pixels number
    pixels_num_sorted = sorted(enumerate(pixels_num), key=lambda x: x[1]) # index is from zero

    return labels, pixels_num_sorted, stats, centroids, contours


# 画ROC曲线
def roc_line(y_score, y_test):
    fpr,tpr,threshold = roc_curve(y_score, y_test,pos_label=1) ###计算真正率和假正率
    roc_auc = auc(fpr,tpr) ###计算auc的值
    # plt.figure()
    # plt.figure(dpi=100, figsize=(3, 3))
    # plt.xticks(fontsize=10)
    # plt.yticks(fontsize=10)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    # plt.show()
    print('AUC: ', roc_auc)
    return fpr,tpr,threshold, roc_auc

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)


    plt.figure(dpi=100, figsize=(3, 3))


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # 改变文字大小参数-fontsize
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted')
    plt.xlabel('True')
    plt.show()


# fpr,tpr,threshold=roc_line(y_score, y_test)
# print(fpr,tpr,threshold)

def hist_similarity(img1, img2):
    import cv2

    H1 = cv2.calcHist([img1], [1], None, [256], [0, 256])
    H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理


    H2 = cv2.calcHist([img2], [1], None, [256], [0, 256])
    H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)


    similarity = cv2.compareHist(H1, H2, 0)

    return similarity


def get_thum(image, size=(64, 64), greyscale=False):
    from PIL import Image
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = Image.fromarray(image)
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image

def image_similarity_vectors_via_numpy(image1, image2):
    import numpy as np
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(np.average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(np.linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = np.dot(a / a_norm, b / b_norm)
    return res