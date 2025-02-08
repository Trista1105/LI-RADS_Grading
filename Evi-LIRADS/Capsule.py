# Update Record

# 1. Change data type accordingly

import array_proc
import numpy as np

import cv2

import FrangiFilter2D
from matplotlib import pyplot as plt

import display


from skimage import morphology


# 20230728 参数自动保存
import argparse
import time
import yaml
import os
import logging

from calculate_candidate_capsule_width import calculate_capsule_width

detailplot_tag = False
small_region_plot_tag = False

logging_tag = False


auto_segmentation = True

use_manual_parameters = False
Capsule_Updated = True

dilation_kernel = 'morphology' # 'cv2', 'morphology'
# use_manual_parameters = False: 基于percentile的自动版本
# use_manual_parameters = True & Capsule_Updated = False：基于手动参数的版本
# use_manual_parameters = True & Capsule_Updated = True：基于手动参数重新优化的版本

# 20230728 参数自动保存
def _parse_args(parser):
    # Do we have a config file to parse?
    # args_config, remaining = config_parser.parse_known_args()
    # if args_config.config:
    #     with open(args_config.config, 'r') as f:
    #         cfg = yaml.safe_load(f)
    #         parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args()

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def capsule_parameters(data_site):
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = parser = argparse.ArgumentParser(description='LIRADS feature Capsule recognition', add_help=False)
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser = argparse.ArgumentParser(description='LIRADS feature Capsule recognition')

    parser.add_argument('--data_site', action='store_true', default=data_site,
                        help='')

    parser.add_argument('--normalize_tag', action='store_true', default=True,
                        help='')

    parser.add_argument('--thresholdingFirstThenRingRegion', action='store_true', default=True,
                        help='# For cases, mask is smaller, not including capsule region')
    if auto_segmentation and (not use_manual_parameters):
        parser.add_argument('--ifDilate', action='store_true', default=True,
                            help='# adding region grow for smaller mask cases, dilation is not needed. Not successful.')
    else:
        if Capsule_Updated:
            parser.add_argument('--ifDilate', action='store_true', default=False,
                                help='# adding region grow for smaller mask cases, dilation is not needed. Not successful.')
        else:
            parser.add_argument('--ifDilate', action='store_true', default=False,
                                help='# adding region grow for smaller mask cases, dilation is not needed. Not successful.')
    parser.add_argument('--Dilate_Size', action='store_true', default=1,
                        help='# dilate size 1.')
    parser.add_argument('--distanceTransformOrErosion', action='store_true', default='erosion',
                        help='# erosion or distanceTransform')
    parser.add_argument('--remove_capsule_within_tumor', action='store_true', default=False,
                        help='')

    if auto_segmentation and (not use_manual_parameters):
        parser.add_argument('--remove_bright_region', action='store_true', default=True,
                            help='')
        parser.add_argument('--remove_bright_liver_ratio', action='store_true', default=95,
                            help=' remove 5%, 用于得到best_roi')
        parser.add_argument('--remove_bright_candidate_capsule_k', action='store_true', default=1.3,
                            help=' 1.3, 用于去除Capsule Bright')

        parser.add_argument('--remove_dark_region', action='store_true', default=True,
                            help='')
        parser.add_argument('--remove_dark_liver_ratio', action='store_true', default=12,
                            help=' remove 12%')
    else:
        if Capsule_Updated:
            parser.add_argument('--remove_dark_region', action='store_true', default=False,
                                help='')
            parser.add_argument('--remove_dark_liver_ratio', action='store_true', default=12,
                                help=' remove 12%')

            parser.add_argument('--remove_bright_region', action='store_true', default=False,
                                help='')
            parser.add_argument('--remove_bright_liver_ratio', action='store_true', default=95,
                                help=' remove 5%, 用于得到best_roi')
            parser.add_argument('--remove_bright_candidate_capsule_k', action='store_true', default=1.3,
                                help=' 1.3, 用于去除Capsule Bright')
        else:
            parser.add_argument('--remove_dark_region', action='store_true', default=False,
                                help='')

            parser.add_argument('--remove_bright_region', action='store_true', default=False,
                            help='')


    parser.add_argument('--remove_candidate_capsule_with_large_width', action='store_true', default=False,
                        help=' ')

    parser.add_argument('--remove_candidate_capsule_with_large_width_threshold', action='store_true', default=2.1,
                        help=' > 3 ')


    parser.add_argument('--weighted_capsule_length', action='store_true', default=False,
                        help='# used in capsule_double_confirm, also in capsule_double_confirm0, tested value: True')




    parser.add_argument('--frangiBetaTwo', action='store_true', default=0.04,
                        help='')

    parser.add_argument('--dilate_disk', action='store_true', default=True,
                        help='')
    parser.add_argument('--upper_lower_around_size', action='store_true', default=1,
                        help='default: 1')

    if auto_segmentation and (not use_manual_parameters):
        parser.add_argument('--score_threshold', action='store_true', default=1.04,
                            help='')

        parser.add_argument('--frangi_threshold_auto', action='store_true', default=False,
                            help='')
    else:
        if Capsule_Updated:
            parser.add_argument('--score_threshold', action='store_true', default=1.04,
                                help='')

            parser.add_argument('--frangi_threshold_auto', action='store_true', default=True,
                                help='')
        else:
            parser.add_argument('--score_threshold', action='store_true', default=1.01,
                                help='')

            parser.add_argument('--frangi_threshold_auto', action='store_true', default=True,
                                help='')

    parser.add_argument('--small_region_k', action='store_true', default=2,
                        help='')


    parser.add_argument('--region_number', action='store_true', default=1,
                        help='# for circular segmentation')
    if auto_segmentation and (not use_manual_parameters):
        parser.add_argument('--separate_long_region', action='store_true', default=True,
                            help='# in order to get accurate upper and lower region around the candidate capsule region')
        parser.add_argument('--long_region_length_k', action='store_true', default=0.25,
                            help='# default: 0.25')
    else:
        if Capsule_Updated:
            parser.add_argument('--separate_long_region', action='store_true', default=True,
                                help='# in order to get accurate upper and lower region around the candidate capsule region')
            parser.add_argument('--long_region_length_k', action='store_true', default=0.5,
                                help='# default: 0.25')
        else:
            parser.add_argument('--separate_long_region', action='store_true', default=False,
                                help='# in order to get accurate upper and lower region around the candidate capsule region')




    if auto_segmentation and (not use_manual_parameters):
        parser.add_argument('--inside_around_gap', action='store_true', default=0,
                            help='')

        parser.add_argument('--erosion_size', action='store_true', default=2,
                            help='# default: 3')
    else:

        if Capsule_Updated:
            parser.add_argument('--inside_around_gap', action='store_true', default=2,
                                help='')

            parser.add_argument('--erosion_size', action='store_true', default=2,
                                help='# default: 3')
        else:
            parser.add_argument('--inside_around_gap', action='store_true', default=2,
                                help='')

            parser.add_argument('--erosion_size', action='store_true', default=3,
                                help='# default: 3')


    if auto_segmentation and (not use_manual_parameters):
        parser.add_argument('--liver_rim_erosion_size', action='store_true', default=4,
                            help='1, 4 tested, default: 4')
        parser.add_argument('--compare_with_candidate_capsule_or_tumor_rim_length', action='store_true', default='compare_with_candidate_capsule',
                            help='# default: compare_with_candidate_capsule, other: compare_with_tumor_rim_length')
        # 出现在remove_capsule_region_along_liver_rim中，每段候选包膜逐一处理

        parser.add_argument('--liver_rim_erosion_capsule_length_threshold', action='store_true', default=0.1,
                            help='')
        parser.add_argument('--erode_candidate_capsule_totally', action='store_true', default=True,
                            help='# default: True')

        #自动版本：每段候选包膜逐一处理
    else:
        parser.add_argument('--liver_rim_erosion_size', action='store_true', default=3,
                            help='1, 4 tested, default: 4')


        #手动版本：所有候选包膜一起处理


    if auto_segmentation and (not use_manual_parameters):
        parser.add_argument('--capsule_upper_lower_liver_eroded', action='store_true', default=True,
                            help='default: True')
        parser.add_argument('--liver_rim_erosion_size_for_capsule', action='store_true', default=0,
                            help='default: 2, 该参数设为0时，相当于capsule_upper_lower_liver_eroded设置成了False')
        parser.add_argument('--capsule_upper_lower_area_ratio', action='store_true', default=0.4,
                            help='# 当包膜位于肝脏边缘时，如果候选包膜两侧的区域太小，或者直接没有肝脏区域，则认为该候选包膜区域不是真正的包膜。'
                                 '该比值是相对于label_sum来说的 \
                            之前手动结果没有该步骤，直接将该参数设置为0')
    else:
        if Capsule_Updated:

            parser.add_argument('--capsule_upper_lower_liver_eroded', action='store_true', default=True,
                                help='default: True')
        else:
            parser.add_argument('--capsule_upper_lower_liver_eroded', action='store_true', default=False,
                                help='default: True')

        parser.add_argument('--liver_rim_erosion_size_for_capsule', action='store_true', default=0,
                            help='default: 2, 该参数设为0时，相当于capsule_upper_lower_liver_eroded设置成了False')
        if Capsule_Updated:
            parser.add_argument('--capsule_upper_lower_area_ratio', action='store_true', default=0,
                                help='# 当包膜位于肝脏边缘时，如果候选包膜两侧的区域太小，或者直接没有肝脏区域，则认为该候选包膜区域不是真正的包膜。'
                                     '该比值是相对于label_sum来说的 \
                                         之前手动结果没有该步骤，直接将该参数设置为0')
        else:
            parser.add_argument('--capsule_upper_lower_area_ratio', action='store_true', default=0,
                                help='# 当包膜位于肝脏边缘时，如果候选包膜两侧的区域太小，或者直接没有肝脏区域，则认为该候选包膜区域不是真正的包膜。'
                                '该比值是相对于label_sum来说的 \
                                    之前手动结果没有该步骤，直接将该参数设置为0')
    # 这部分出现在capsule_double_confirm0中，erosion部分和上面liver_rim_erosion部分看起来是重复了，最终版本该部分没有erosion，但包膜两侧的面积大小还是作为一个限制条件被采用了。

    if auto_segmentation and (not use_manual_parameters):
        parser.add_argument('--statisticType', action='store_true', default='percentile',
                            help='default: True')
    else:
        if Capsule_Updated:
            parser.add_argument('--statisticType', action='store_true', default='percentile',
                                help='default: True')
        else:
            parser.add_argument('--statisticType', action='store_true', default='mean',
                                help='default: True')


    args, args_text = _parse_args(parser)

    if not args.normalize_tag:
        parser.add_argument('--deviation_threshold_max', action='store_true', default=40,
                            help='')

        parser.add_argument('--deviation_threshold_min', action='store_true', default=15,
                            help='')

    else:
        parser.add_argument('--deviation_threshold_max', action='store_true', default=15,
                            help='')

        parser.add_argument('--deviation_threshold_min', action='store_true', default=5,
                            help='')


    args, args_text = _parse_args(parser)

    return parser, args, args_text




def tumor_distance_transform(t_label_win, inside_threshold, radius, dilate_k, outside_threshold, args):
    if args.distanceTransformOrErosion == 'distanceTransform':
        # Distance transform for lesion/tumor
        dist = cv2.distanceTransform(src=t_label_win.astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
        # dist1 = cv2.convertScaleAbs(dist)
        dist2 = cv2.normalize(dist, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # y_close, x_close = np.where(1-dist2/255 < 0.8)[0], np.where(1-dist2/255 < 0.8)[1]
        # heat_img = cv2.applyColorMap(dist2, cv2.COLORMAP_JET)

        y_close_dis_inside, x_close_dis_inside = np.where(1 - dist2 / 255 < inside_threshold)[0], \
                                                 np.where(1 - dist2 / 255 < inside_threshold)[1]

    if args.distanceTransformOrErosion == 'erosion':
        if args.erosion_size > 0:
            t_label_win_erosion = array_proc.erosion(t_label_win, args.erosion_size)
        else:
            t_label_win_erosion = t_label_win

        y_close_dis_inside, x_close_dis_inside = np.where(t_label_win_erosion == 1)[0], \
                                                   np.where(t_label_win_erosion == 1)[1]
    # y_close_dis_outside, x_close_dis_outside = np.where(dist2_inverse / 255 >= 0.2)[0], \
    #                            np.where(dist2_inverse / 255 >= 0.2)[1]
    if args.ifDilate:

        t_label_win_dilate = array_proc.dilate_size(t_label_win, args.Dilate_Size) # Test 1 on 20230522

    else:
        t_label_win_dilate = t_label_win

    y_close_dis_outside, x_close_dis_outside = np.where(t_label_win_dilate == 0)[0], \
                                               np.where(t_label_win_dilate == 0)[1]




    return y_close_dis_inside, x_close_dis_inside, y_close_dis_outside, x_close_dis_outside

def get_lesion_around(dyn_2d_win, t_label_win, l_2d_win, radius):
    import copy
    if dilation_kernel == 'cv2':
        lesion_around = array_proc.dilate(t_label_win, np.int0(radius))
    if dilation_kernel == 'morphology':
        lesion_around = array_proc.dilate_morphologyKernel_cv2Dilation(t_label_win, np.int0(radius))

    y, x = np.where(lesion_around == 0)
    dyn_2d_win_lesion_around = copy.copy(dyn_2d_win)
    dyn_2d_win_lesion_around[y, x] = 0


    y, x = np.where(l_2d_win == 0)
    dyn_2d_win_lesion_around[y, x] = 0



    return dyn_2d_win_lesion_around


def padding(dyn_2d_win_input):
    import copy
    dyn_2d_win = copy.deepcopy(dyn_2d_win_input)

    y_before_padding, x_before_padding = np.where(dyn_2d_win == 0)[0], np.where(dyn_2d_win == 0)[1]

    row_num = dyn_2d_win.shape[0]
    col_num = dyn_2d_win.shape[1]


    for row in range(0, row_num):
        x = []
        for col in range(0, col_num):
            if dyn_2d_win[row, col] > 0:
                x.append(col)

        if len(x) >= 1:
            x_left = min(x)
            x_right = max(x)

            # dyn_2d_win[row, 0:x_left + 1] = np.mean(cv2.blur(dyn_2d_win[row-3:row+3, x_left:x_left+5], (3,3)))
            # dyn_2d_win[row, x_right - 1:col_num] = np.mean(cv2.blur(dyn_2d_win[row-3:row+3, x_right-5:x_right], (3,3)))

            if x_left < round(col_num / 2):
                dyn_2d_win[row, 0:x_left] = dyn_2d_win[row, x_left + 1:(2 * x_left + 1)][::-1]
            else:
                dyn_2d_win[row, 2 * x_left - col_num + 1:x_left] = dyn_2d_win[row, x_left + 1:col_num][::-1]

            if x_right > round(col_num / 2) - 1:
                dyn_2d_win[row, x_right:col_num - 1] = dyn_2d_win[row, 2 * x_right - col_num + 1:x_right][::-1]
            else:
                if x_right >= 1:
                    dyn_2d_win[row, x_right + 1:(2 * x_right)] = dyn_2d_win[row, 0:x_right - 1]

            # dyn_2d_win[row, 0:x_left + 1] = np.mean(dyn_2d_win)
            # dyn_2d_win[row, x_right - 1:col_num] = np.mean(dyn_2d_win)


    return dyn_2d_win, y_before_padding, x_before_padding


def paddingHV0(dyn_2d_win_input):
    import copy
    dyn_2d_win = copy.copy(dyn_2d_win_input)

    y_before_padding, x_before_padding = np.where(dyn_2d_win == 0)

    row_num, col_num = dyn_2d_win.shape


    for k in range(0, 3):

        for row in range(0, row_num):
            x = []
            for col in range(0, col_num):
                if dyn_2d_win[row, col] > 0:
                    x.append(col)

            if len(x) >= 1:
                x_left = min(x)
                x_right = max(x)

                # dyn_2d_win[row, 0:x_left + 1] = np.mean(cv2.blur(dyn_2d_win[row-3:row+3, x_left:x_left+5], (3,3)))
                # dyn_2d_win[row, x_right - 1:col_num] = np.mean(cv2.blur(dyn_2d_win[row-3:row+3, x_right-5:x_right], (3,3)))

                if x_left < round(col_num / 2):
                    dyn_2d_win[row, 0:x_left] = dyn_2d_win[row, x_left + 1:(2 * x_left + 1)][::-1]
                else:
                    dyn_2d_win[row, 2 * x_left - col_num + 1:x_left] = dyn_2d_win[row, x_left + 1:col_num][::-1]

                if x_right > round(col_num / 2) - 1:
                    dyn_2d_win[row, x_right:col_num - 1] = dyn_2d_win[row, 2 * x_right - col_num + 1:x_right][::-1]
                else:
                    if x_right >= 1:
                        dyn_2d_win[row, x_right + 1:(2 * x_right)] = dyn_2d_win[row, 0:x_right - 1]

                # dyn_2d_win[row, 0:x_left + 1] = np.mean(dyn_2d_win)
                # dyn_2d_win[row, x_right - 1:col_num] = np.mean(dyn_2d_win)



        for col in range(0, col_num):

            y = []
            for row in range(0, row_num):
                if dyn_2d_win[row, col] > 0:
                    y.append(row)

            if len(y) >= 1:
                y_upper = min(y)
                y_lower = max(y)

                if y_upper < round(row_num / 2) and y_lower >= 2 * y_upper + 1:
                    for row in range(0, y_upper+1):
                        dyn_2d_win[row, col] = dyn_2d_win[2*y_upper-row+1, col]

                if (y_upper < round(row_num / 2) and y_lower < 2 * y_upper + 1) or y_upper >= round(row_num / 2):
                    for row in range(y_upper + 1, y_lower+1):
                        dyn_2d_win[2*y_upper-row+1, col] = dyn_2d_win[row, col]

                if y_lower > round(row_num / 2) - 1:
                    dyn_2d_win[y_lower:row_num - 1, col] = dyn_2d_win[2 * y_lower - row_num + 1:y_lower, col]
                else:
                    if y_lower >= 1:
                        dyn_2d_win[y_lower + 1:(2 * y_lower), col] = dyn_2d_win[0:y_lower - 1, col]



    for col in range(0, col_num):

        for row in range(0, row_num):
            if dyn_2d_win[row, col] == 0:
                dyn_2d_win[row, col] = dyn_2d_win[row, col-1]

    for row in range(0, row_num):

        for col in range(0, col_num):
            if dyn_2d_win[row, col] == 0:
                dyn_2d_win[row, col] = dyn_2d_win[row-1, col]




    return dyn_2d_win, y_before_padding, x_before_padding


def paddingHV(dyn_2d_win_input):
    dyn_2d_win = copy.copy(dyn_2d_win_input)
    y_before_padding, x_before_padding = np.where(dyn_2d_win == 0)
    row_num, col_num = dyn_2d_win.shape

    for _ in range(3):
        # 水平方向处理
        for row in range(row_num):
            valid_cols = np.where(dyn_2d_win[row] > 0)[0]
            if valid_cols.size > 0:
                x_left, x_right = valid_cols[0], valid_cols[-1]

                if x_left < round(col_num / 2):
                    dyn_2d_win[row, 0:x_left] = dyn_2d_win[row, x_left + 1:(2 * x_left + 1)][::-1]
                else:
                    dyn_2d_win[row, 2 * x_left - col_num + 1:x_left] = dyn_2d_win[row, x_left + 1:col_num][::-1]

                if x_right > round(col_num / 2) - 1:
                    dyn_2d_win[row, x_right:col_num - 1] = dyn_2d_win[row, 2 * x_right - col_num + 1:x_right][::-1]
                else:
                    if x_right >= 1:
                        dyn_2d_win[row, x_right + 1:(2 * x_right)] = dyn_2d_win[row, 0:x_right - 1]

        # 垂直方向处理
        for col in range(col_num):
            valid_rows = np.where(dyn_2d_win[:, col] > 0)[0]
            if valid_rows.size > 0:
                y_upper, y_lower = valid_rows[0], valid_rows[-1]

                if y_upper < round(row_num / 2) and y_lower >= 2 * y_upper + 1:
                    for row in range(0, y_upper + 1):
                        dyn_2d_win[row, col] = dyn_2d_win[2 * y_upper - row + 1, col]

                if (y_upper < round(row_num / 2) and y_lower < 2 * y_upper + 1) or y_upper >= round(row_num / 2):
                    for row in range(y_upper + 1, y_lower + 1):
                        dyn_2d_win[2 * y_upper - row + 1, col] = dyn_2d_win[row, col]

                if y_lower > round(row_num / 2) - 1:
                    dyn_2d_win[y_lower:row_num - 1, col] = dyn_2d_win[2 * y_lower - row_num + 1:y_lower, col]
                else:
                    if y_lower >= 1:
                        dyn_2d_win[y_lower + 1:(2 * y_lower), col] = dyn_2d_win[0:y_lower - 1, col]

    for col in range(col_num):
        for row in range(0, row_num):
            if dyn_2d_win[row, col] == 0:
                dyn_2d_win[row, col] = dyn_2d_win[row, col - 1]

    for row in range(0, row_num):
        for col in range(0, col_num):
            if dyn_2d_win[row, col] == 0:
                dyn_2d_win[row, col] = dyn_2d_win[row - 1, col]

    return dyn_2d_win, y_before_padding, x_before_padding




def frangi_thresholding(dyn_2d_win_enhanced, frangi_threshold, radius, args):
    import copy
    frangi_thresholded = copy.deepcopy(dyn_2d_win_enhanced)
    cv2.normalize(dyn_2d_win_enhanced, frangi_thresholded, 0, 255, cv2.NORM_MINMAX)
    if args.frangi_threshold_auto:
        frangi_threshold = 2 * int(np.floor(0.5 * (dyn_2d_win_enhanced.shape[0] + dyn_2d_win_enhanced.shape[1]) / 16)) + 1

    frangi_thresholded = cv2.adaptiveThreshold(frangi_thresholded.astype(np.uint8), 255,
                                               cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, frangi_threshold,
                                               0)  # 75

    frangi_thresholded_large = array_proc.remove_small_t(frangi_thresholded, np.int0(0.1 * radius))  # 15

    return frangi_thresholded, frangi_thresholded_large

def frangi_thresholding_false_capsule_based(dyn_2d_win_enhanced, t_label_win, l_2d_win, radius, args):
    import copy
    frangi_thresholded = copy.deepcopy(dyn_2d_win_enhanced)
    cv2.normalize(dyn_2d_win_enhanced, frangi_thresholded, 0, 255, cv2.NORM_MINMAX)
    # frangi_thresholded = cv2.adaptiveThreshold(frangi_thresholded.astype(np.uint8), 255,
    #                                            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, frangi_threshold,
    #                                            0)  # 75
    frangi_thresholded_around = copy.deepcopy(frangi_thresholded)

    t_label_win_dilate = array_proc.dilate_size(t_label_win, radius)
    y,x = np.where(t_label_win == 1)[0], np.where(t_label_win == 1)[1]
    t_label_win_dilate[y,x] = 0
    y, x = np.where(t_label_win_dilate == 0)[0], np.where(t_label_win_dilate == 0)[1]
    frangi_thresholded_around[y, x] = 0

    y,x = np.where(l_2d_win == 0)[0], np.where(l_2d_win == 0)[1]
    frangi_thresholded_around[y, x] = 0

    around_max_intensity = np.max(frangi_thresholded_around)

    # print(around_max_intensity)


    y,x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
    frangi_thresholded[y,x] = 0

    # plt.figure()
    # plt.subplot(131)
    # plt.imshow(frangi_thresholded_around, cmap='gray')
    # plt.subplot(132)
    # plt.imshow(frangi_thresholded, cmap='gray')
    if args.data_site == 'ZheYi':
        ret, frangi_thresholded = cv2.threshold(frangi_thresholded.astype(np.uint8), 0.5*around_max_intensity, 255, cv2.THRESH_BINARY)  # 160
        frangi_thresholded_large = array_proc.remove_small_t(frangi_thresholded, np.int0(0.1 * radius))  # 15
    if args.data_site == 'ZhongShan' or args.data_site == 'SuZhou'or args.data_site == 'PHC':
        ret, frangi_thresholded = cv2.threshold(frangi_thresholded.astype(np.uint8), 0.5*around_max_intensity, 255,
                                                cv2.THRESH_BINARY)  # 160
        frangi_thresholded_large = array_proc.remove_small_t(frangi_thresholded, np.int0(0.1 * radius))  # 15

    # plt.subplot(133)
    # plt.imshow(frangi_thresholded, cmap='gray')
    # plt.show()

    return frangi_thresholded, frangi_thresholded_large

def get_capsule_region_no_padding(dyn_2d_win,y_close_dis_inside, x_close_dis_inside, y_close_dis_outside, x_close_dis_outside):
    import copy
    dyn_2d_win_enhanced = copy.deepcopy(dyn_2d_win)
    dyn_2d_win_enhanced[y_close_dis_outside, x_close_dis_outside] = 0
    dyn_2d_win_enhanced[y_close_dis_inside, x_close_dis_inside] = 0

    return dyn_2d_win_enhanced

def get_capsule_region_padding(dyn_2d_win,y_close_dis_inside, x_close_dis_inside, y_close_dis_outside, x_close_dis_outside, y_before_padding, x_before_padding):
    import copy
    dyn_2d_win_enhanced = copy.deepcopy(dyn_2d_win)

    dyn_2d_win_enhanced[y_before_padding, x_before_padding] = 0

    dyn_2d_win_enhanced[y_close_dis_outside, x_close_dis_outside] = 0
    dyn_2d_win_enhanced[y_close_dis_inside, x_close_dis_inside] = 0

    return dyn_2d_win_enhanced

def capsule_region_thresholding(capsule_region, radius):
    adaptive_threshold_mask_tumor = 35
    import copy
    dyn_2d_win_enhanced = copy.deepcopy(capsule_region)
    cv2.normalize(capsule_region, dyn_2d_win_enhanced, 0, 255, cv2.NORM_MINMAX)
    kernel_size = min(adaptive_threshold_mask_tumor, 2 * np.int0(0.5 * radius) + 1)
    if kernel_size == 1:
        kernel_size = 3
    capsule_region_thresholded = cv2.adaptiveThreshold(dyn_2d_win_enhanced.astype(np.uint8), 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                   kernel_size, 0)  # small kernel, detect more

    capsule_region_thresholded_large = array_proc.remove_small_t(capsule_region_thresholded, np.int0(0.2 * radius))  # 15

    return capsule_region_thresholded, capsule_region_thresholded_large

def capsule_region_thresholding_OTSU(capsule_region, radius):
    adaptive_threshold_mask_tumor = 35
    import copy
    dyn_2d_win_enhanced = copy.deepcopy(capsule_region)
    cv2.normalize(capsule_region, dyn_2d_win_enhanced, 0, 255, cv2.NORM_MINMAX)
    # kernel_size = min(adaptive_threshold_mask_tumor, 2 * np.int0(0.5 * radius) + 1)
    # if kernel_size == 1:
    #     kernel_size = 3
    # capsule_region_thresholded = cv2.adaptiveThreshold(dyn_2d_win_enhanced.astype(np.uint8), 255,
    #                                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
    #                                kernel_size, 0)  # small kernel, detect more

    ret, capsule_region_thresholded = cv2.threshold(dyn_2d_win_enhanced.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    capsule_region_thresholded_large = array_proc.remove_small_t(capsule_region_thresholded, np.int0(0.2 * radius))  # 15

    return capsule_region_thresholded, capsule_region_thresholded_large

def get_capsule_region_threshouded_around(capsule_region_thresholded_large):
    if dilation_kernel == 'cv2':
        capsule_region_threshouded_around = array_proc.dilate(capsule_region_thresholded_large, 2)
    if dilation_kernel == 'morphology':
        capsule_region_threshouded_around = array_proc.dilate_morphologyKernel_cv2Dilation(capsule_region_thresholded_large, 2)

    y, x = np.where(capsule_region_thresholded_large > 0)[0], np.where(capsule_region_thresholded_large > 0)[1]
    capsule_region_threshouded_around[y, x] = 0

    return capsule_region_threshouded_around



def lesion_ring_skeleton(dyn_2d_win, t_label_win):
    dyn_2d_win_skeleton = copy.deepcopy(dyn_2d_win)

    y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
    dyn_2d_win_skeleton[y, x] = 0

    t_label_win_erosion_1 = array_proc.erosion(t_label_win, 1)
    y, x = np.where(t_label_win_erosion_1 == 1)[0], np.where(t_label_win_erosion_1 == 1)[1]

    dyn_2d_win_skeleton[y, x] = 0

    y, x = np.where(dyn_2d_win_skeleton > 0)[0], np.where(dyn_2d_win_skeleton > 0)[1]

    dyn_2d_win_skeleton[y, x] = 1

    return dyn_2d_win_skeleton

def get_capsule_region_skeleton(capsule_region):
    from skimage import morphology
    capsule_region_skeleton = morphology.skeletonize(capsule_region)

    return capsule_region_skeleton


def img_gradient(img, t_label_win, radius):
    import cv2
    import numpy as np

    sobelx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1)
    sobely = cv2.convertScaleAbs(sobely)
    result = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    t_label_win_dilated = array_proc.dilate(t_label_win, min(np.int0(0.2*radius), 1))
    y, x = np.where(t_label_win_dilated == 0)[0], np.where(t_label_win_dilated == 0)[1]
    result[y, x] = 0

    # plt.figure()
    # plt.imshow(result, cmap = 'gray')
    # plt.show()

    return result

def img_gradient_ROI(img, t_label_win, radius):
    import cv2
    import numpy as np

    sobelx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1)
    sobely = cv2.convertScaleAbs(sobely)
    result = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

    # t_label_win_dilated = array_proc.dilate(t_label_win, min(np.int0(0.2*radius), 1))
    # y, x = np.where(t_label_win_dilated == 0)[0], np.where(t_label_win_dilated == 0)[1]
    # result[y, x] = 0

    # plt.figure()
    # plt.imshow(result, cmap = 'gray')
    # plt.show()

    return result

def liver_vessel_removal_ROI(dyn_2d_win, t_label_win, l_2d_win, radius, plotfigure_tag):
    dilate_size = 3
    # remove blood vessel

    # remove very dark area within the liver around tumor
    dyn_2d_win_normalized = copy.deepcopy(dyn_2d_win)
    cv2.normalize(dyn_2d_win, dyn_2d_win_normalized, 0, 255, cv2.NORM_MINMAX)
    # result = cv2.adaptiveThreshold(dyn_2d_win_normalized.astype(np.uint8), 255,
    #                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
    #                                       2 * np.int0(0.5 * min(dyn_2d_win.shape[0], dyn_2d_win.shape[1])) + 1,
    #                                       0)  # min(35, 2*np.int0(0.5*radius)+1)
    ret, result = cv2.threshold(dyn_2d_win_normalized.astype(np.uint8), 140, 255, cv2.THRESH_BINARY)  # 140, #100
    result = array_proc.erosion(result, 1)
    result = array_proc.dilate_size(result, 1)
    result = array_proc.remove_small_t(result,
                                       0.3 * min(dyn_2d_win.shape[0], dyn_2d_win.shape[1]) * min(dyn_2d_win.shape[0],
                                                                                                 dyn_2d_win.shape[1]))
    result = array_proc.hole_fill(result)
    # avoid tumor area
    y, x = np.where(t_label_win == 1)[0], np.where(t_label_win == 1)[1]
    result[y, x] = 0

    y, x = np.where(result == 0)[0], np.where(result == 0)[1]
    dyn_2d_win_remove_dark = copy.deepcopy(dyn_2d_win)
    dyn_2d_win_remove_dark[y, x] = 0



    if plotfigure_tag:
        plt.figure()

        dyn_2d_win_liver_contour = copy.deepcopy(dyn_2d_win)
        dyn_2d_win_liver_contour = display.add_contour_to_img(dyn_2d_win_liver_contour, l_2d_win, 1,
                                                              (0, 255, 0),
                                                              1)

        plt.subplot(3, 6, 1)
        plt.imshow(dyn_2d_win, cmap='gray')
        plt.title('dyn_2d_win')

        plt.subplot(3, 6, 2)
        plt.imshow(dyn_2d_win_liver_contour, cmap='gray')
        plt.title('dyn_2d_win_liver_contour')

        plt.subplot(3, 6, 3)
        plt.imshow(result, cmap='gray')
        plt.title('dyn_2d_win_remove_vessel')

        plt.subplot(3, 6, 4)
        plt.imshow(dyn_2d_win_remove_dark, cmap='gray')
        plt.title('dyn_2d_win_remove_dark')

        plt.show()

def liver_vessel_removal(dyn_2d, t_label, l_2d, radius, plotfigure_tag, vessel_removal):

    # remove blood vessel

    # Get liver region
    liver_region = copy.deepcopy(dyn_2d)
    y,x = np.where(l_2d == 0)[0], np.where(l_2d == 0)[1]
    liver_region[y,x] = 0

    dyn_2d_normalized = copy.deepcopy(liver_region)
    cv2.normalize(liver_region, dyn_2d_normalized, 0, 255, cv2.NORM_MINMAX)

    # threshTwoPeaks(dyn_2d_normalized.astype(np.uint8))

    # result = cv2.adaptiveThreshold(dyn_2d_normalized.astype(np.uint8), 255,
    #                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
    #                                       min(31, 2 * np.int0(0.5 * min(dyn_2d.shape[0], dyn_2d.shape[1])) + 1),
    #                                       0)

    ret, result = cv2.threshold(dyn_2d_normalized.astype(np.uint8), 160, 255, cv2.THRESH_BINARY) #160
    # ret, result = cv2.threshold(dyn_2d_normalized.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret, result = cv2.threshold(dyn_2d_normalized.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    result_large = copy.deepcopy(result)
    result_large = array_proc.erosion(result_large, 1)
    result_large = array_proc.dilate_size(result_large, 1)
    result_large = array_proc.remove_small_t(result_large, 20)

    exist = (result_large != 0)
    large_sum = exist.sum()

    # result = array_proc.hole_fill(result)

    # remove very large region, which is obviously not vessel
    labels, sort_list, stats, centroids, contours = array_proc.region_selection(result_large)
    for region_index in range(1, labels.max() + 1):

        label = copy.deepcopy(labels)
        y, x = np.where(labels != region_index)
        label[y, x] = 0
        y, x = np.where(labels == region_index)
        label[y, x] = 1

        exist = (label != 0)
        label_sum = exist.sum()

        if label_sum > 0.1*large_sum:
            labels[y,x] = 0

    y,x = np.where(labels != 0)[0], np.where(labels != 0)[1]
    labels[y,x] = 1


    # # avoid tumor area
    # y, x = np.where(t_label == 1)[0], np.where(t_label == 1)[1]
    # result[y, x] = 0

    # remove very dark area within the liver around tumor
    # y, x = np.where(result == 0)[0], np.where(result == 0)[1]
    # dyn_2d_remove_dark = copy.deepcopy(dyn_2d)
    # dyn_2d_remove_dark[y, x] = 0

    # plotfigure_tag = False
    if plotfigure_tag and vessel_removal:
        plt.figure()

        dyn_2d_liver_contour = copy.deepcopy(dyn_2d)
        dyn_2d_liver_contour = display.add_contour_to_img(dyn_2d_liver_contour, l_2d, 1,
                                                              (0, 255, 0),
                                                              1)



        plt.subplot(2, 2, 1)
        plt.imshow(dyn_2d_liver_contour, cmap='gray')
        plt.title('dyn_2d_win_liver_contour')

        plt.subplot(2, 2, 2)
        plt.imshow(dyn_2d_normalized, cmap='gray')
        plt.title('liver_region_normalized')

        plt.subplot(2, 2, 3)
        plt.imshow(result, cmap='gray')
        plt.title('dyn_2d_win_remove_vessel')

        result_large_contour = display.add_contour_to_img(liver_region, labels, 1,
                                                              (0, 255, 0),
                                                              1)

        result_large_contour = display.add_contour_to_img(result_large_contour, t_label, 0,
                                                              (255, 0, 0),
                                                              1)

        plt.subplot(2, 2, 4)
        plt.imshow(result_large_contour, cmap='gray')
        plt.title('dyn_2d_win_remove_vessel_large')

        # plt.subplot(2, 2, 4)
        # plt.imshow(dyn_2d_remove_dark, cmap='gray')
        # plt.title('dyn_2d_win_remove_dark')

        plt.show()

    return result_large, labels

def compare_both_sides_upper_lower(dyn_2d_win_copy, t_label_win_, labels, label, region_index, upper_lower_around_size, lesion_rim_sum, lesion_around_contour, region_label, remove_tumor_dark, args):
    label_dilate = array_proc.dilate(label, upper_lower_around_size)

    y, x = np.where(label == region_index)[0], np.where(label == region_index)[1]
    label_dilate[y, x] = 0

    # remove very dark region around capsule region
    if remove_tumor_dark:
        y, x = np.where(t_label_win_ == 0)[0], np.where(t_label_win_ == 0)[1]
        label_dilate[y,x] = 0

    # Get upper and lower edge of the capsule region
    y, x = np.where(label == 1)[0], np.where(label == 1)[1]

    label_left_x = min(x)
    label_right_x = max(x)

    X = []
    Y_max = []
    Y_min = []
    for x_value in range(label_left_x, label_right_x + 1):
        X.append(x_value)
        x_index = np.argwhere(x == x_value)
        y_max = min(y)
        y_min = max(y)
        for index in range(0, len(x_index)):
            y_index = x_index[index][0]
            y_value = y[y_index]
            if y_value > y_max:
                y_max = y_value
            if y_value < y_min:
                y_min = y_value

        Y_max.append(y_max)
        Y_min.append(y_min)

    # End Get upper and lower edge of the capsule region

    # Seperate upper and lower part of the capsule around region
    upper_y = []
    upper_x = []
    lower_y = []
    lower_x = []

    all_y = []
    all_x = []

    y_label_dilate, x_label_dilate = np.where(label_dilate == 1)[0], \
                                     np.where(label_dilate == 1)[1]

    for x_value_index in range(0, len(X)):
        x_value = X[x_value_index]
        y_max = Y_max[x_value_index]
        y_min = Y_min[x_value_index]

        x_index = np.argwhere(x_label_dilate == x_value)

        for index in range(0, len(x_index)):
            y_index = x_index[index][0]
            y_value = y_label_dilate[y_index]

            if y_value > y_max + args.inside_around_gap:
                lower_x.append(x_value)
                lower_y.append(y_value)

                all_x.append(x_value)
                all_y.append(y_value)

            if y_value < y_min - args.inside_around_gap:
                upper_x.append(x_value)
                upper_y.append(y_value)

                all_x.append(x_value)
                all_y.append(y_value)

    mask = np.zeros((labels.shape[0], labels.shape[1]))
    mask[all_y, all_x] = 1
    y, x = np.where(mask == 0)[0], np.where(mask == 0)[1]

    label_dilate_upper = copy.deepcopy(label_dilate)
    label_dilate_upper[y, x] = 0
    label_dilate_upper[lower_y, lower_x] = 0

    label_dilate_lower = copy.deepcopy(label_dilate)
    label_dilate_lower[y, x] = 0
    label_dilate_lower[upper_y, upper_x] = 0

    # End Seperate upper and lower part of the capsule around region


    # result = display.add_contour_to_img(dyn_2d_win_copy, region_label, 1,
    #                                                  (0, 0, 255), 1)
    # result = display.add_contour_to_img(result, label_dilate_upper, 0,
    #                                           (255, 0, 0), 1)
    # result = display.add_contour_to_img(result, label_dilate_lower, 0,
    #                                     (0, 255, 0), 1)


    if args.normalize_tag:
        cv2.normalize(dyn_2d_win_copy, dyn_2d_win_copy, 0, 255, cv2.NORM_MINMAX)
    y, x = np.where(label == 0)[0], np.where(label == 0)[1]
    region_label_ROI = copy.deepcopy(dyn_2d_win_copy)
    region_label_ROI[y, x] = 0

    y, x = np.where(label_dilate_upper == 0)[0], np.where(label_dilate_upper == 0)[1]
    region_around_upper = copy.deepcopy(dyn_2d_win_copy)
    region_around_upper[y, x] = 0

    y, x = np.where(label_dilate_lower == 0)[0], np.where(label_dilate_lower == 0)[1]
    region_around_lower = copy.deepcopy(dyn_2d_win_copy)
    region_around_lower[y, x] = 0


    # separate label region into small regions
    y,x = np.where(region_label_ROI)[0], np.where(region_label_ROI)[1]
    y_l = np.where(region_around_lower)[0]
    y_u = np.where(region_around_upper)[0]

    x_left = min(x)
    x_right = max(x)

    length_x = max(x)-min(x) + 1
    num_small_region = np.int0(length_x/(args.small_region_k*lesion_rim_sum)) + 1


    if small_region_plot_tag:
        plt.figure()

        plt.subplot(5, 3, 1)
        plt.imshow(dyn_2d_win_copy, cmap='gray')
        plt.title('ROI')

        plt.subplot(5, 3, 2)
        plt.imshow(lesion_around_contour, cmap='gray')
        plt.title('ROI with label')

        plt.subplot(5, 3, 3)
        plt.imshow(region_label, cmap='gray')
        plt.title('region_label')

        plt.subplot(5, 3, 4)
        plt.imshow(region_label_ROI, cmap='gray')
        plt.subplot(5, 3, 5)
        plt.imshow(region_around_lower, cmap='gray')
        plt.subplot(5, 3, 6)
        plt.imshow(region_around_upper, cmap='gray')

    Upper_score = []
    Lower_score = []
    small_capsule_tag = []
    small_capsule_length = []
    for index in range(0, num_small_region):

        mask = np.zeros((region_label_ROI.shape[0], region_label_ROI.shape[1]), np.uint8)
        mask[0:region_label_ROI.shape[0], x_left+index*np.int0(args.small_region_k*lesion_rim_sum):x_left+(index+1)*np.int0(args.small_region_k*lesion_rim_sum)] = 255

        y,x = np.where(mask == 0)[0], np.where(mask == 0)[1]

        region_label_ROI_small = copy.deepcopy(region_label_ROI)
        region_label_ROI_small[y,x] = 0

        region_around_lower_small = copy.deepcopy(region_around_lower)
        region_around_lower_small[y,x] = 0

        region_around_upper_small = copy.deepcopy(region_around_upper)
        region_around_upper_small[y, x] = 0

        exist_inside = (region_label_ROI_small != 0)
        inside_mean = region_label_ROI_small.sum() / exist_inside.sum()

        exist = (region_around_upper_small != 0)
        around_upper_mean = region_around_upper_small.sum() / exist.sum()

        exist = (region_around_lower_small != 0)
        around_lower_mean = region_around_lower_small.sum() / exist.sum()

        upper_score = round(inside_mean / around_upper_mean, 2)
        lower_score = round(inside_mean / around_lower_mean, 2)

        Upper_score.append(upper_score)
        Lower_score.append(lower_score)

        label_skeleton = get_capsule_region_skeleton(region_label_ROI_small > 0)
        exist = (label_skeleton != 0)
        small_capsule_length.append(exist.sum())

        if ((upper_score >= args.score_threshold) and (lower_score >= args.score_threshold)):
            small_capsule_tag.append(min(upper_score, lower_score))
        else:
            small_capsule_tag.append(0)

        if small_region_plot_tag:
            plt.subplot(7, 3, 3*(index+2)+1)
            plt.imshow(region_label_ROI_small, cmap='gray')
            plt.text(5, 20, 'inside mean: ' + str(round(inside_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.subplot(7, 3, 3 * (index + 2) + 2)
            plt.imshow(region_around_lower_small, cmap='gray')
            plt.text(5, 20, 'lower mean: ' + str(round(around_lower_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 40, 'lower score: ' + str(round(lower_score, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            plt.subplot(7, 3, 3 * (index + 2) + 3)
            plt.imshow(region_around_upper_small, cmap='gray')
            plt.text(5, 20, 'upper mean: ' + str(round(around_upper_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 40, 'upper score: ' + str(round(upper_score, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

    if small_region_plot_tag:
        plt.show()

    # calculate capsule length
    capsule_sum = 0
    # small_capsule_tag_ = np.array(small_capsule_tag)
    # capsule_tag = np.int64(small_capsule_tag_ > 0)
    # capsule_tag = capsule_tag.tolist()
    for cap_index in range(0, len(small_capsule_tag)):
        if small_capsule_tag[cap_index] > 0:
            # label = copy.deepcopy(labels)
            # y, x = np.where(labels != cap_index + 1)
            # label[y, x] = 0
            # y, x = np.where(labels == cap_index + 1)
            # label[y, x] = 1
            # label_skeleton = get_capsule_region_skeleton(label)
            # label_skeleton = label_skeleton.astype(int)
            # exist = (label_skeleton != 0)
            #
            # capsule_sum = capsule_sum + exist.sum() * capsule_tag[cap_index]
            capsule_sum = capsule_sum + small_capsule_length[cap_index]

    return Upper_score, Lower_score, capsule_sum, label_dilate, \
           X, Y_max, Y_min, region_label_ROI, region_around_lower, region_around_upper

def compare_both_sides_left_right(dyn_2d_win_copy, t_label_win_, labels, label, region_index, upper_lower_around_size, lesion_rim_sum, lesion_around_contour, region_label, remove_tumor_dark, args):
    label_dilate = array_proc.dilate(label, upper_lower_around_size)

    y, x = np.where(label == region_index)[0], np.where(label == region_index)[1]
    label_dilate[y, x] = 0

    # remove very dark region around capsule region
    if remove_tumor_dark:
        y, x = np.where(t_label_win_ == 0)[0], np.where(t_label_win_ == 0)[1]
        label_dilate[y,x] = 0

    # Get left and right edge of the capsule region
    y, x = np.where(label == 1)[0], np.where(label == 1)[1]

    label_lower_y = min(y)
    label_upper_y = max(y)

    Y = []
    X_max = []
    X_min = []
    for y_value in range(label_lower_y, label_upper_y + 1):
        Y.append(y_value)
        y_index = np.argwhere(y == y_value)
        x_max = min(x)
        x_min = max(x)
        for index in range(0, len(y_index)):
            x_index = y_index[index][0]
            x_value = x[x_index]
            if x_value > x_max:
                x_max = x_value
            if x_value < x_min:
                x_min = x_value

        X_max.append(x_max)
        X_min.append(x_min)

    # End Get left and right edge of the capsule region

    # Seperate left and right part of the capsule around region
    right_x = []
    right_y = []
    left_x = []
    left_y = []

    all_x = []
    all_y = []

    y_label_dilate, x_label_dilate = np.where(label_dilate == 1)[0], \
                                     np.where(label_dilate == 1)[1]

    for y_value_index in range(0, len(Y)):
        y_value = Y[y_value_index]
        x_max = X_max[y_value_index]
        x_min = X_min[y_value_index]

        y_index = np.argwhere(y_label_dilate == y_value)

        for index in range(0, len(y_index)):
            x_index = y_index[index][0]
            x_value = x_label_dilate[x_index]

            if x_value > x_max + args.inside_around_gap:
                left_x.append(x_value)
                left_y.append(y_value)

                all_x.append(x_value)
                all_y.append(y_value)

            if x_value < x_min - args.inside_around_gap:
                right_x.append(x_value)
                right_y.append(y_value)

                all_x.append(x_value)
                all_y.append(y_value)

    mask = np.zeros((labels.shape[0], labels.shape[1]))
    mask[all_y, all_x] = 1
    y, x = np.where(mask == 0)[0], np.where(mask == 0)[1]

    label_dilate_right = copy.deepcopy(label_dilate)
    label_dilate_right[y, x] = 0
    label_dilate_right[left_y, left_x] = 0

    label_dilate_left = copy.deepcopy(label_dilate)
    label_dilate_left[y, x] = 0
    label_dilate_left[right_y, right_x] = 0

    # End Seperate upper and lower part of the capsule around region
    # result = display.add_contour_to_img(dyn_2d_win_copy, region_label, 1,
    #                                                  (0, 0, 255), 1)
    # result = display.add_contour_to_img(result, label_dilate_upper, 0,
    #                                           (255, 0, 0), 1)
    # result = display.add_contour_to_img(result, label_dilate_lower, 0,
    #                                     (0, 255, 0), 1)

    # plt.subplot(3, 6, 16)
    # plt.imshow(mask, cmap='gray')
    # plt.subplot(3, 6, 17)
    # plt.imshow(label_dilate_upper, cmap='gray')
    # plt.subplot(3, 6, 18)
    # plt.imshow(label_dilate_lower, cmap='gray')

    if args.normalize_tag:
        cv2.normalize(dyn_2d_win_copy, dyn_2d_win_copy, 0, 255, cv2.NORM_MINMAX)
    y, x = np.where(label == 0)[0], np.where(label == 0)[1]
    region_label_ROI = copy.deepcopy(dyn_2d_win_copy)
    region_label_ROI[y, x] = 0

    y, x = np.where(label_dilate_right == 0)[0], np.where(label_dilate_right == 0)[1]
    region_around_right = copy.deepcopy(dyn_2d_win_copy)
    region_around_right[y, x] = 0

    y, x = np.where(label_dilate_left == 0)[0], np.where(label_dilate_left == 0)[1]
    region_around_left = copy.deepcopy(dyn_2d_win_copy)
    region_around_left[y, x] = 0

    # separate label region into small regions
    y,x = np.where(region_label_ROI)[0], np.where(region_label_ROI)[1]
    x_l = np.where(region_around_left)[1]
    x_r = np.where(region_around_right)[1]

    y_lower = min(y)
    y_upper = max(y)

    length_y = max(y)-min(y) + 1
    num_small_region = np.int0(length_y/(args.small_region_k*lesion_rim_sum)) + 1


    if small_region_plot_tag:
        plt.figure()

        plt.subplot(5, 3, 1)
        plt.imshow(dyn_2d_win_copy, cmap='gray')
        plt.title('ROI')

        plt.subplot(5, 3, 2)
        plt.imshow(lesion_around_contour, cmap='gray')
        plt.title('ROI with label')

        plt.subplot(5, 3, 3)
        plt.imshow(region_label, cmap='gray')
        plt.title('region_label')

        plt.subplot(5, 3, 4)
        plt.imshow(region_label_ROI, cmap='gray')
        plt.subplot(5, 3, 5)
        plt.imshow(region_around_left, cmap='gray')
        plt.subplot(5, 3, 6)
        plt.imshow(region_around_right, cmap='gray')

    Upper_score = []
    Lower_score = []
    small_capsule_tag = []
    small_capsule_length = []
    for index in range(0, num_small_region):

        mask = np.zeros((region_label_ROI.shape[0], region_label_ROI.shape[1]), np.uint8)
        # mask[0:region_label_ROI.shape[0], x_left+index*np.int0(0.1*lesion_rim_sum):x_left+(index+1)*np.int0(0.1*lesion_rim_sum)] = 255
        mask[y_lower + index * np.int0(args.small_region_k * lesion_rim_sum):y_lower + (index + 1) * np.int0(args.small_region_k * lesion_rim_sum),
        0:region_label_ROI.shape[1]] = 255

        y,x = np.where(mask == 0)[0], np.where(mask == 0)[1]

        region_label_ROI_small = copy.deepcopy(region_label_ROI)
        region_label_ROI_small[y,x] = 0

        region_around_lower_small = copy.deepcopy(region_around_left)
        region_around_lower_small[y,x] = 0

        region_around_upper_small = copy.deepcopy(region_around_right)
        region_around_upper_small[y, x] = 0

        exist_inside = (region_label_ROI_small != 0)
        inside_mean = region_label_ROI_small.sum() / exist_inside.sum()

        exist = (region_around_upper_small != 0)
        around_upper_mean = region_around_upper_small.sum() / exist.sum()

        exist = (region_around_lower_small != 0)
        around_lower_mean = region_around_lower_small.sum() / exist.sum()

        upper_score = round(inside_mean / around_upper_mean, 2)
        lower_score = round(inside_mean / around_lower_mean, 2)

        Upper_score.append(upper_score)
        Lower_score.append(lower_score)

        label_skeleton = get_capsule_region_skeleton(region_label_ROI_small > 0)
        exist = (label_skeleton != 0)
        small_capsule_length.append(exist.sum())


        if ((upper_score >= args.score_threshold) and (lower_score >= args.score_threshold)):
            small_capsule_tag.append(min(upper_score, lower_score))

        else:
            small_capsule_tag.append(0)

        if small_region_plot_tag:
            plt.subplot(7, 3, 3*(index+2)+1)
            plt.imshow(region_label_ROI_small, cmap='gray')
            plt.text(5, 20, 'inside mean: ' + str(round(inside_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.subplot(7, 3, 3 * (index + 2) + 2)
            plt.imshow(region_around_lower_small, cmap='gray')
            plt.text(5, 20, 'lower mean: ' + str(round(around_lower_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 40, 'lower score: ' + str(round(lower_score, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            plt.subplot(7, 3, 3 * (index + 2) + 3)
            plt.imshow(region_around_upper_small, cmap='gray')
            plt.text(5, 20, 'upper mean: ' + str(round(around_upper_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 40, 'upper score: ' + str(round(upper_score, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

    if small_region_plot_tag:
        plt.show()

    # calculate capsule length
    capsule_sum = 0
    # small_capsule_tag_ = np.array(small_capsule_tag)
    # capsule_tag = np.int64(small_capsule_tag_ > 0)
    # capsule_tag = capsule_tag.tolist()
    for cap_index in range(0, len(small_capsule_tag)):
        if small_capsule_tag[cap_index] > 0:
            # label = copy.deepcopy(labels)
            # y, x = np.where(labels != cap_index + 1)
            # label[y, x] = 0
            # y, x = np.where(labels == cap_index + 1)
            # label[y, x] = 1
            # label_skeleton = get_capsule_region_skeleton(label)
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(label, cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(label_skeleton, cmap='gray')
            # plt.show()
            # exist = (label_skeleton != 0)
            # capsule_sum = capsule_sum + exist.sum() * capsule_tag[cap_index]
            capsule_sum = capsule_sum + small_capsule_length[cap_index]

    return Upper_score, Lower_score, capsule_sum, label_dilate, Y, X_max, X_min, \
           region_label_ROI, region_around_left, region_around_right


def compare_both_sides(dyn_2d_win_copy, t_label_win_, labels, label, region_index, direction, upper_lower_around_size,
                                  lesion_rim_sum, lesion_around_contour, region_label, remove_tumor_dark,args):
    if dilation_kernel == 'cv2':
        label_dilate = array_proc.dilate(label, upper_lower_around_size)
    if dilation_kernel == 'morphology':
        label_dilate = array_proc.dilate_morphologyKernel_cv2Dilation(label, upper_lower_around_size)

    y, x = np.where(label == region_index)[0], np.where(label == region_index)[1]
    label_dilate[y, x] = 0

    # remove very dark region around capsule region
    if remove_tumor_dark:
        y, x = np.where(t_label_win_ == 0)[0], np.where(t_label_win_ == 0)[1]
        label_dilate[y, x] = 0

    end_x = []
    end_y = []
    start_x = []
    start_y = []

    all_x = []
    all_y = []

    if direction == 'V':
        # Get left and right edge of the capsule region
        y, x = np.where(label == 1)[0], np.where(label == 1)[1]

        label_lower_y = min(y)
        label_upper_y = max(y)

        Y = []
        X_max = []
        X_min = []
        for y_value in range(label_lower_y, label_upper_y + 1):
            Y.append(y_value)
            y_index = np.argwhere(y == y_value)
            x_max = min(x)
            x_min = max(x)
            for index in range(0, len(y_index)):
                x_index = y_index[index][0]
                x_value = x[x_index]
                if x_value > x_max:
                    x_max = x_value
                if x_value < x_min:
                    x_min = x_value

            X_max.append(x_max)
            X_min.append(x_min)

        # End Get left and right edge of the capsule region

    # Seperate left and right part of the capsule around region

        y_label_dilate, x_label_dilate = np.where(label_dilate == 1)[0], \
                                         np.where(label_dilate == 1)[1]

        for y_value_index in range(0, len(Y)):
            y_value = Y[y_value_index]
            x_max = X_max[y_value_index]
            x_min = X_min[y_value_index]

            y_index = np.argwhere(y_label_dilate == y_value)

            for index in range(0, len(y_index)):
                x_index = y_index[index][0]
                x_value = x_label_dilate[x_index]

                if x_value > x_max + args.inside_around_gap:
                    start_x.append(x_value)
                    start_y.append(y_value)

                    all_x.append(x_value)
                    all_y.append(y_value)

                if x_value < x_min - args.inside_around_gap:
                    end_x.append(x_value)
                    end_y.append(y_value)

                    all_x.append(x_value)
                    all_y.append(y_value)
    else:
        # Get upper and lower edge of the capsule region
        y, x = np.where(label == 1)[0], np.where(label == 1)[1]

        label_left_x = min(x)
        label_right_x = max(x)

        X = []
        Y_max = []
        Y_min = []
        for x_value in range(label_left_x, label_right_x + 1):
            X.append(x_value)
            x_index = np.argwhere(x == x_value)
            y_max = min(y)
            y_min = max(y)
            for index in range(0, len(x_index)):
                y_index = x_index[index][0]
                y_value = y[y_index]
                if y_value > y_max:
                    y_max = y_value
                if y_value < y_min:
                    y_min = y_value

            Y_max.append(y_max)
            Y_min.append(y_min)

        # End Get upper and lower edge of the capsule region

        # Seperate upper and lower part of the capsule around region
        y_label_dilate, x_label_dilate = np.where(label_dilate == 1)[0], \
                                         np.where(label_dilate == 1)[1]

        for x_value_index in range(0, len(X)):
            x_value = X[x_value_index]
            y_max = Y_max[x_value_index]
            y_min = Y_min[x_value_index]

            x_index = np.argwhere(x_label_dilate == x_value)

            for index in range(0, len(x_index)):
                y_index = x_index[index][0]
                y_value = y_label_dilate[y_index]

                if y_value > y_max + args.inside_around_gap:
                    start_x.append(x_value)
                    start_y.append(y_value)

                    all_x.append(x_value)
                    all_y.append(y_value)

                if y_value < y_min - args.inside_around_gap:
                    end_x.append(x_value)
                    end_y.append(y_value)

                    all_x.append(x_value)
                    all_y.append(y_value)

    mask = np.zeros((labels.shape[0], labels.shape[1]))
    mask[all_y, all_x] = 1
    y, x = np.where(mask == 0)[0], np.where(mask == 0)[1]

    label_dilate_end = copy.deepcopy(label_dilate)
    label_dilate_end[y, x] = 0
    label_dilate_end[start_y, start_x] = 0

    label_dilate_start = copy.deepcopy(label_dilate)
    label_dilate_start[y, x] = 0
    label_dilate_start[end_y, end_x] = 0

    # End Seperate left and right part of the capsule around region


    if args.normalize_tag:
        cv2.normalize(dyn_2d_win_copy, dyn_2d_win_copy, 0, 255, cv2.NORM_MINMAX)
    y, x = np.where(label == 0)[0], np.where(label == 0)[1]
    region_label_ROI = copy.deepcopy(dyn_2d_win_copy)
    region_label_ROI[y, x] = 0

    y, x = np.where(label_dilate_end == 0)[0], np.where(label_dilate_end == 0)[1]
    region_around_end = copy.deepcopy(dyn_2d_win_copy)
    region_around_end[y, x] = 0

    y, x = np.where(label_dilate_start == 0)[0], np.where(label_dilate_start == 0)[1]
    region_around_start = copy.deepcopy(dyn_2d_win_copy)
    region_around_start[y, x] = 0

    # separate label region into small regions
    y, x = np.where(region_label_ROI)[0], np.where(region_label_ROI)[1]

    if direction == 'V':
        y_lower = min(y)
        length_y = max(y) - min(y) + 1
        num_small_region = np.int0(length_y / (args.small_region_k * lesion_rim_sum)) + 1
    else:
        x_left = min(x)
        length_x = max(x) - min(x) + 1
        num_small_region = np.int0(length_x / (args.small_region_k * lesion_rim_sum)) + 1

    if small_region_plot_tag:
        plt.figure()

        plt.subplot(5, 3, 1)
        plt.imshow(dyn_2d_win_copy, cmap='gray')
        plt.title('ROI')

        plt.subplot(5, 3, 2)
        plt.imshow(lesion_around_contour, cmap='gray')
        plt.title('ROI with label')

        plt.subplot(5, 3, 3)
        plt.imshow(region_label, cmap='gray')
        plt.title('region_label')

        plt.subplot(5, 3, 4)
        plt.imshow(region_label_ROI, cmap='gray')
        plt.subplot(5, 3, 5)
        plt.imshow(region_around_start, cmap='gray')
        plt.subplot(5, 3, 6)
        plt.imshow(region_around_end, cmap='gray')

    Upper_score = []
    Lower_score = []
    small_capsule_tag = []
    small_capsule_length = []
    for index in range(0, num_small_region):

        mask = np.zeros((region_label_ROI.shape[0], region_label_ROI.shape[1]), np.uint8)
        if direction == 'V':
            mask[y_lower + index * np.int0(args.small_region_k * lesion_rim_sum):y_lower + (index + 1) * np.int0(
                args.small_region_k * lesion_rim_sum),0:region_label_ROI.shape[1]] = 255
        else:
            mask[0:region_label_ROI.shape[0],
            x_left + index * np.int0(args.small_region_k * lesion_rim_sum):x_left + (index + 1) * np.int0(
                args.small_region_k * lesion_rim_sum)] = 255

        y, x = np.where(mask == 0)[0], np.where(mask == 0)[1]

        region_label_ROI_small = copy.deepcopy(region_label_ROI)
        region_label_ROI_small[y, x] = 0

        region_around_lower_small = copy.deepcopy(region_around_start)
        region_around_lower_small[y, x] = 0

        region_around_upper_small = copy.deepcopy(region_around_end)
        region_around_upper_small[y, x] = 0

        exist_inside = (region_label_ROI_small != 0)
        inside_mean = region_label_ROI_small.sum() / exist_inside.sum()

        exist = (region_around_upper_small != 0)
        around_upper_mean = region_around_upper_small.sum() / exist.sum()

        exist = (region_around_lower_small != 0)
        around_lower_mean = region_around_lower_small.sum() / exist.sum()

        upper_score = round(inside_mean / around_upper_mean, 2)
        lower_score = round(inside_mean / around_lower_mean, 2)

        Upper_score.append(upper_score)
        Lower_score.append(lower_score)

        label_skeleton = get_capsule_region_skeleton(region_label_ROI_small > 0)
        exist = (label_skeleton != 0)
        small_capsule_length.append(exist.sum())

        if ((upper_score >= args.score_threshold) and (lower_score >= args.score_threshold)):
            small_capsule_tag.append(min(upper_score, lower_score))

        else:
            small_capsule_tag.append(0)

        if small_region_plot_tag:
            plt.subplot(7, 3, 3 * (index + 2) + 1)
            plt.imshow(region_label_ROI_small, cmap='gray')
            plt.text(5, 20, 'inside mean: ' + str(round(inside_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.subplot(7, 3, 3 * (index + 2) + 2)
            plt.imshow(region_around_lower_small, cmap='gray')
            plt.text(5, 20, 'lower mean: ' + str(round(around_lower_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 40, 'lower score: ' + str(round(lower_score, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            plt.subplot(7, 3, 3 * (index + 2) + 3)
            plt.imshow(region_around_upper_small, cmap='gray')
            plt.text(5, 20, 'upper mean: ' + str(round(around_upper_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 40, 'upper score: ' + str(round(upper_score, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))


            plt.subplot(7, 3, 3 * (index + 3) + 1)
            plt.imshow(region_label_ROI_small, cmap='gray')

            plt.subplot(7, 3, 3 * (index + 3) + 2)
            plt.imshow(region_around_lower_small, cmap='gray')

            plt.subplot(7, 3, 3 * (index + 3) + 3)
            plt.imshow(region_around_upper_small, cmap='gray')


    if small_region_plot_tag:
        plt.show()

    # calculate capsule length
    capsule_sum = 0
    # small_capsule_tag_ = np.array(small_capsule_tag)
    # capsule_tag = np.int64(small_capsule_tag_ > 0)
    # capsule_tag = capsule_tag.tolist()
    for cap_index in range(0, len(small_capsule_tag)):
        if small_capsule_tag[cap_index] > 0:
            # label = copy.deepcopy(labels)
            # y, x = np.where(labels != cap_index + 1)
            # label[y, x] = 0
            # y, x = np.where(labels == cap_index + 1)
            # label[y, x] = 1
            # label_skeleton = get_capsule_region_skeleton(label)
            # # plt.figure()
            # # plt.subplot(1,2,1)
            # # plt.imshow(label, cmap='gray')
            # # plt.subplot(1, 2, 2)
            # # plt.imshow(label_skeleton, cmap='gray')
            # # plt.show()
            # exist = (label_skeleton != 0)
            # capsule_sum = capsule_sum + exist.sum() * capsule_tag[cap_index]
            if args.weighted_capsule_length:
                capsule_sum = capsule_sum + small_capsule_length[cap_index]*small_capsule_tag[cap_index]
            else:
                capsule_sum = capsule_sum + small_capsule_length[cap_index]
    if direction == 'V':
        return Upper_score, Lower_score, capsule_sum, label_dilate, Y, X_max, X_min, \
               region_label_ROI, region_around_start, region_around_end
    else:
        return Upper_score, Lower_score, capsule_sum, label_dilate, \
               X, Y_max, Y_min, region_label_ROI, region_around_start, region_around_end

def compare_both_sides_circular_sub(dyn_2d_win_copy, t_label_win_, labels, label, region_index, direction, upper_lower_around_size,
                                  lesion_rim_sum, lesion_around_contour, region_label, remove_tumor_dark, args):
    label_dilate = array_proc.dilate(label, upper_lower_around_size)

    y, x = np.where(label == region_index)[0], np.where(label == region_index)[1]
    label_dilate[y, x] = 0

    # remove very dark region around capsule region
    if remove_tumor_dark:
        y, x = np.where(t_label_win_ == 0)[0], np.where(t_label_win_ == 0)[1]
        label_dilate[y, x] = 0

    end_x = []
    end_y = []
    start_x = []
    start_y = []

    all_x = []
    all_y = []

    if direction == 'V':
        # Get left and right edge of the capsule region
        y, x = np.where(label == 1)[0], np.where(label == 1)[1]

        label_lower_y = min(y)
        label_upper_y = max(y)

        Y = []
        X_max = []
        X_min = []
        for y_value in range(label_lower_y, label_upper_y + 1):
            Y.append(y_value)
            y_index = np.argwhere(y == y_value)
            x_max = min(x)
            x_min = max(x)
            for index in range(0, len(y_index)):
                x_index = y_index[index][0]
                x_value = x[x_index]
                if x_value > x_max:
                    x_max = x_value
                if x_value < x_min:
                    x_min = x_value

            X_max.append(x_max)
            X_min.append(x_min)

        # End Get left and right edge of the capsule region

    # Seperate left and right part of the capsule around region

        y_label_dilate, x_label_dilate = np.where(label_dilate == 1)[0], \
                                         np.where(label_dilate == 1)[1]

        for y_value_index in range(0, len(Y)):
            y_value = Y[y_value_index]
            x_max = X_max[y_value_index]
            x_min = X_min[y_value_index]

            y_index = np.argwhere(y_label_dilate == y_value)

            for index in range(0, len(y_index)):
                x_index = y_index[index][0]
                x_value = x_label_dilate[x_index]

                if x_value > x_max + args.inside_around_gap:
                    start_x.append(x_value)
                    start_y.append(y_value)

                    all_x.append(x_value)
                    all_y.append(y_value)

                if x_value < x_min - args.inside_around_gap:
                    end_x.append(x_value)
                    end_y.append(y_value)

                    all_x.append(x_value)
                    all_y.append(y_value)
    else:
        # Get upper and lower edge of the capsule region
        y, x = np.where(label == 1)[0], np.where(label == 1)[1]

        label_left_x = min(x)
        label_right_x = max(x)

        X = []
        Y_max = []
        Y_min = []
        for x_value in range(label_left_x, label_right_x + 1):
            X.append(x_value)
            x_index = np.argwhere(x == x_value)
            y_max = min(y)
            y_min = max(y)
            for index in range(0, len(x_index)):
                y_index = x_index[index][0]
                y_value = y[y_index]
                if y_value > y_max:
                    y_max = y_value
                if y_value < y_min:
                    y_min = y_value

            Y_max.append(y_max)
            Y_min.append(y_min)

        # End Get upper and lower edge of the capsule region

        # Seperate upper and lower part of the capsule around region
        y_label_dilate, x_label_dilate = np.where(label_dilate == 1)[0], \
                                         np.where(label_dilate == 1)[1]

        for x_value_index in range(0, len(X)):
            x_value = X[x_value_index]
            y_max = Y_max[x_value_index]
            y_min = Y_min[x_value_index]

            x_index = np.argwhere(x_label_dilate == x_value)

            for index in range(0, len(x_index)):
                y_index = x_index[index][0]
                y_value = y_label_dilate[y_index]

                if y_value > y_max + args.inside_around_gap:
                    start_x.append(x_value)
                    start_y.append(y_value)

                    all_x.append(x_value)
                    all_y.append(y_value)

                if y_value < y_min - args.inside_around_gap:
                    end_x.append(x_value)
                    end_y.append(y_value)

                    all_x.append(x_value)
                    all_y.append(y_value)

    mask = np.zeros((labels.shape[0], labels.shape[1]))
    mask[all_y, all_x] = 1
    y, x = np.where(mask == 0)[0], np.where(mask == 0)[1]

    label_dilate_end = copy.deepcopy(label_dilate)
    label_dilate_end[y, x] = 0
    label_dilate_end[start_y, start_x] = 0

    label_dilate_start = copy.deepcopy(label_dilate)
    label_dilate_start[y, x] = 0
    label_dilate_start[end_y, end_x] = 0

    # End Seperate left and right part of the capsule around region


    if args.normalize_tag:
        cv2.normalize(dyn_2d_win_copy, dyn_2d_win_copy, 0, 255, cv2.NORM_MINMAX)
    y, x = np.where(label == 0)[0], np.where(label == 0)[1]
    region_label_ROI = copy.deepcopy(dyn_2d_win_copy)
    region_label_ROI[y, x] = 0

    y, x = np.where(label_dilate_end == 0)[0], np.where(label_dilate_end == 0)[1]
    region_around_end = copy.deepcopy(dyn_2d_win_copy)
    region_around_end[y, x] = 0

    y, x = np.where(label_dilate_start == 0)[0], np.where(label_dilate_start == 0)[1]
    region_around_start = copy.deepcopy(dyn_2d_win_copy)
    region_around_start[y, x] = 0

    # separate label region into small regions
    y, x = np.where(region_label_ROI)[0], np.where(region_label_ROI)[1]

    if direction == 'V':
        y_lower = min(y)
        length_y = max(y) - min(y) + 1
        num_small_region = np.int0(length_y / (args.small_region_k * lesion_rim_sum)) + 1
    else:
        x_left = min(x)
        length_x = max(x) - min(x) + 1
        num_small_region = np.int0(length_x / (args.small_region_k * lesion_rim_sum)) + 1

    if small_region_plot_tag:
        plt.figure()

        plt.subplot(5, 3, 1)
        plt.imshow(dyn_2d_win_copy, cmap='gray')
        plt.title('ROI')

        plt.subplot(5, 3, 2)
        plt.imshow(lesion_around_contour, cmap='gray')
        plt.title('ROI with label')

        plt.subplot(5, 3, 3)
        plt.imshow(region_label, cmap='gray')
        plt.title('region_label')

        plt.subplot(5, 3, 4)
        plt.imshow(region_label_ROI, cmap='gray')
        plt.subplot(5, 3, 5)
        plt.imshow(region_around_start, cmap='gray')
        plt.subplot(5, 3, 6)
        plt.imshow(region_around_end, cmap='gray')

    Upper_score = []
    Lower_score = []
    small_capsule_tag = []
    small_capsule_length = []
    for index in range(0, num_small_region):

        mask = np.zeros((region_label_ROI.shape[0], region_label_ROI.shape[1]), np.uint8)
        if direction == 'V':
            mask[y_lower + index * np.int0(args.small_region_k * lesion_rim_sum):y_lower + (index + 1) * np.int0(
                args.small_region_k * lesion_rim_sum),0:region_label_ROI.shape[1]] = 255
        else:
            mask[0:region_label_ROI.shape[0],
            x_left + index * np.int0(args.small_region_k * lesion_rim_sum):x_left + (index + 1) * np.int0(
                args.small_region_k * lesion_rim_sum)] = 255

        y, x = np.where(mask == 0)[0], np.where(mask == 0)[1]

        region_label_ROI_small = copy.deepcopy(region_label_ROI)
        region_label_ROI_small[y, x] = 0

        region_around_lower_small = copy.deepcopy(region_around_start)
        region_around_lower_small[y, x] = 0

        region_around_upper_small = copy.deepcopy(region_around_end)
        region_around_upper_small[y, x] = 0

        exist_inside = (region_label_ROI_small != 0)
        inside_mean = region_label_ROI_small.sum() / exist_inside.sum()

        exist = (region_around_upper_small != 0)
        around_upper_mean = region_around_upper_small.sum() / exist.sum()

        exist = (region_around_lower_small != 0)
        around_lower_mean = region_around_lower_small.sum() / exist.sum()

        upper_score = round(inside_mean / around_upper_mean, 2)
        lower_score = round(inside_mean / around_lower_mean, 2)

        Upper_score.append(upper_score)
        Lower_score.append(lower_score)

        label_skeleton = get_capsule_region_skeleton(region_label_ROI_small > 0)
        exist = (label_skeleton != 0)
        small_capsule_length.append(exist.sum())

        if ((upper_score >= args.score_threshold) and (lower_score >= args.score_threshold)):
            small_capsule_tag.append(min(upper_score, lower_score))

        else:
            small_capsule_tag.append(0)

        if small_region_plot_tag:
            plt.subplot(7, 3, 3 * (index + 2) + 1)
            plt.imshow(region_label_ROI_small, cmap='gray')
            plt.text(5, 20, 'inside mean: ' + str(round(inside_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.subplot(7, 3, 3 * (index + 2) + 2)
            plt.imshow(region_around_lower_small, cmap='gray')
            plt.text(5, 20, 'lower mean: ' + str(round(around_lower_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 40, 'lower score: ' + str(round(lower_score, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

            plt.subplot(7, 3, 3 * (index + 2) + 3)
            plt.imshow(region_around_upper_small, cmap='gray')
            plt.text(5, 20, 'upper mean: ' + str(round(around_upper_mean, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
            plt.text(5, 40, 'upper score: ' + str(round(upper_score, 2)),
                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

    if small_region_plot_tag:
        plt.show()

    # calculate capsule length
    capsule_sum = 0
    # small_capsule_tag_ = np.array(small_capsule_tag)
    # capsule_tag = np.int64(small_capsule_tag_ > 0)
    # capsule_tag = capsule_tag.tolist()
    for cap_index in range(0, len(small_capsule_tag)):
        if small_capsule_tag[cap_index] > 0:
            # label = copy.deepcopy(labels)
            # y, x = np.where(labels != cap_index + 1)
            # label[y, x] = 0
            # y, x = np.where(labels == cap_index + 1)
            # label[y, x] = 1
            # label_skeleton = get_capsule_region_skeleton(label)
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(label, cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(label_skeleton, cmap='gray')
            # plt.show()
            # exist = (label_skeleton != 0)
            # capsule_sum = capsule_sum + exist.sum() * capsule_tag[cap_index]
            if args.weighted_capsule_length:
                capsule_sum = capsule_sum + small_capsule_length[cap_index]*small_capsule_tag[cap_index]
            else:
                capsule_sum = capsule_sum + small_capsule_length[cap_index]

    if direction == 'V':
        return capsule_sum, Y, X_max, X_min
    else:
        return capsule_sum, X, Y_max, Y_min


def compare_both_sides_circular(dyn_2d_win_copy, t_label_win_, regions, labels, label, region_index, upper_lower_around_size,
                                  lesion_rim_sum, lesion_around_contour, region_label, remove_tumor_dark, args):
    Capsule_sum = 0
    for i in range(0, args.region_number):
        labels_ = copy.deepcopy(labels)
        label_ = copy.deepcopy(label)


        y, x = np.where(regions != i)[0], np.where(regions != i)[1]
        labels_[y,x] = 0
        label_[y,x] = 0

        label_ = array_proc.remove_small_t(label_, 2)

        if (label_ != 0).sum() > 0:

            direction = label_direction(label_)

            # print(i+1, direction)

            capsule_sum, Y, X_max, X_min = compare_both_sides_circular_sub(dyn_2d_win_copy, t_label_win_, labels_, label_, region_index, direction, upper_lower_around_size,
                                      lesion_rim_sum, lesion_around_contour, region_label, remove_tumor_dark,args)

            Capsule_sum = Capsule_sum + capsule_sum

    return Capsule_sum

def capsule_double_confirm(capsule_region_thresholded_large, dyn_2d_win, t_label_win, t_label_win_, logging_tag, plotfigure_tag, remove_tumor_dark, args):
    dyn_2d_win_copy = copy.deepcopy(dyn_2d_win)


    labels, sort_list, stats, centroids, contours = array_proc.region_selection(capsule_region_thresholded_large)

    lesion_around_contour = display.add_contour_to_img(dyn_2d_win_copy, capsule_region_thresholded_large, 1,
                                                       (0, 0, 255), 1)

    num_region = labels.max()

    if logging_tag:
        print('Total region num is: ', num_region)

    # calculate lesion rim length
    dyn_2d_win_skeleton = copy.deepcopy(dyn_2d_win_copy)

    y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
    dyn_2d_win_skeleton[y, x] = 0

    t_label_win_erosion_1 = array_proc.erosion(t_label_win, 1)
    y, x = np.where(t_label_win_erosion_1 == 1)[0], np.where(t_label_win_erosion_1 == 1)[1]

    dyn_2d_win_skeleton[y, x] = 0

    y, x = np.where(dyn_2d_win_skeleton > 0)[0], np.where(dyn_2d_win_skeleton > 0)[1]

    dyn_2d_win_skeleton[y, x] = 1

    exist = (dyn_2d_win_skeleton != 0)
    lesion_rim_sum = exist.sum()
    # End calculate lesion rim length


    # capsule_tag = []
    Capsule_sum = 0
    for region_index in range(1, num_region + 1):

        label = copy.deepcopy(labels)
        y, x = np.where(labels != region_index)
        label[y, x] = 0
        y, x = np.where(labels == region_index)
        label[y, x] = 1

        exist = (label != 0)
        label_sum = exist.sum()

        if label_sum >= 0.05*lesion_rim_sum:


            if logging_tag:
                print('For region: ', region_index)


            region_label = display.add_contour_to_img(dyn_2d_win_copy, label, 1,
                                                      (0, 255, 0), 1)

            # calculate direction of the label
            # label_copy = copy.deepcopy(label)
            # img = np.uint8(label_copy)
            #
            # ret, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)
            # contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # # minAreaRect = cv2.minAreaRect(contours[0])
            # boundingRect = cv2.boundingRect(contours[0])
            #
            #
            # if boundingRect[2]/boundingRect[3] < 1:
            #     direction = 'V'
            # else:
            #     direction = 'H'
            direction = label_direction(label)

            # Compare with both sides
            if args.data_site == 'ZhongShan' or args.data_site == 'SuZhou' or args.data_site == 'PHC':
                # if direction == 'H':
                #     Upper_score, Lower_score, capsule_sum, label_dilate, X, Y_max, Y_min, \
                #     region_label_ROI, region_around_lower, region_around_upper = compare_both_sides_upper_lower(dyn_2d_win_copy, t_label_win_, labels, label, region_index,
                #                                                              upper_lower_around_size, lesion_rim_sum, lesion_around_contour, region_label, remove_tumor_dark=remove_tumor_dark)
                # if direction == 'V':
                #     Upper_score, Lower_score, capsule_sum, label_dilate, Y, X_max, X_min, \
                #     region_label_ROI, region_around_lower, region_around_upper = compare_both_sides_left_right(dyn_2d_win_copy, t_label_win_, labels, label, region_index, upper_lower_around_size, lesion_rim_sum, lesion_around_contour, region_label, remove_tumor_dark=remove_tumor_dark)
                #

                Upper_score, Lower_score, capsule_sum, label_dilate, Y, X_max, X_min, \
                region_label_ROI, region_around_lower, region_around_upper = compare_both_sides(dyn_2d_win_copy,
                                                                                                           t_label_win_,
                                                                                                           labels, label,
                                                                                                           region_index,
                                                                                                           direction,
                                                                                                           args.upper_lower_around_size,
                                                                                                           lesion_rim_sum,
                                                                                                           lesion_around_contour,
                                                                                                           region_label,
                                                                                                           remove_tumor_dark=remove_tumor_dark,args=args)

                # regions = mask_segment(dyn_2d_win, args.region_number, args)
                # capsule_sum = compare_both_sides_circular(dyn_2d_win_copy,
                #                                            t_label_win_,
                #                                            regions,
                #                                            labels, label,
                #                                            region_index,
                #                                            args.upper_lower_around_size,
                #                                            lesion_rim_sum,
                #                                            lesion_around_contour,
                #                                            region_label,
                #                                            remove_tumor_dark=remove_tumor_dark,
                #                                           args=args)


                # if ((upper_score >= score_threshold) and (lower_score >= score_threshold)):
                #     if logging_tag:
                #         print('region ', region_index, ' is capsule region!')
                #     capsule_tag.append(min(upper_score, lower_score))
                # else:
                #     capsule_tag.append(0)
                #
                # # calculate capsule length
                # capsule_sum = 0
                # capsule_tag = np.int64(capsule_tag > 0)
                # for cap_index in range(0, len(capsule_tag)):
                #     if capsule_tag[cap_index] > 0:
                #         label = copy.deepcopy(labels)
                #         y,x = np.where(labels != cap_index + 1)
                #         label[y,x] = 0
                #         y, x = np.where(labels == cap_index + 1)
                #         label[y, x] = 1
                #         label_skeleton = get_capsule_region_skeleton(label)
                #         exist = (label_skeleton != 0)
                #         capsule_sum = capsule_sum + exist.sum()*capsule_tag[cap_index]

                # capsule_sum = capsule_sum + exist.sum() * (1 + 10 * (capsule_tag[cap_index] - 1))
                # End calculate capsule length

                # End compare with both sides

            if args.data_site == 'ZheYi':
                regions = mask_segment(dyn_2d_win, args.region_number,args)
                # plt.figure()
                # plt.imshow(regions, cmap='gray')
                # plt.show()

                # Upper_score, Lower_score, capsule_sum, label_dilate, Y, X_max, X_min, \
                # region_label_ROI, region_around_lower, region_around_upper = compare_both_sides_circular(dyn_2d_win_copy,
                #                                                                                            t_label_win_,
                #                                                                                            regions,
                #                                                                                            labels, label,
                #                                                                                            region_index,
                #                                                                                            upper_lower_around_size,
                #                                                                                            lesion_rim_sum,
                #                                                                                            lesion_around_contour,
                #                                                                                            region_label,
                #                                                                                            remove_tumor_dark=remove_tumor_dark)

                capsule_sum = compare_both_sides_circular(dyn_2d_win_copy,
                                                           t_label_win_,
                                                           regions,
                                                           labels, label,
                                                           region_index,
                                                           args.upper_lower_around_size,
                                                           lesion_rim_sum,
                                                           lesion_around_contour,
                                                           region_label,
                                                           remove_tumor_dark=remove_tumor_dark,
                                                          args=args)



            Capsule_sum = Capsule_sum + capsule_sum


            if capsule_sum > 0:
                if detailplot_tag:
                    plt.figure()
                    plt.subplot(3, 6, 1)
                    plt.imshow(dyn_2d_win_copy, cmap='gray')
                    plt.title('ROI')

                    plt.subplot(3, 6, 2)
                    plt.imshow(lesion_around_contour, cmap='gray')
                    plt.title('ROI with label')
                if detailplot_tag:
                    plt.subplot(3, 6, 3)
                    plt.imshow(region_label, cmap='gray')
                    plt.title('region_label')

                # if detailplot_tag:
                #     plt.subplot(3, 6, 4)
                #     plt.imshow(label_dilate, cmap='gray')
                #     plt.title('label_dilate')

                # label_skeleton = morphology.skeletonize(label)
                #
                # region_label_skeleton = display.add_contour_to_img(region_label, label_skeleton, 0,
                #                                           (0, 0, 255), 1)
                if detailplot_tag:
                    plt.subplot(3, 6, 5)
                    plt.imshow(region_label, cmap='gray')
                    plt.title('region_label')
                # if detailplot_tag:
                #     if direction == 'H':
                #         for col in range(0, len(Y)):
                #             plt.scatter(Y[col], X_max[col], color='orange')
                #             plt.scatter(Y[col], X_min[col], color='purple')
                #     if direction == 'V':
                #         for row in range(0, len(Y)):
                #             plt.scatter(X_max[row], Y[row], color='orange')
                #             plt.scatter(X_min[row], Y[row], color='purple')

                # if detailplot_tag:
                #     plt.subplot(3, 6, 7)
                #     plt.imshow(region_label_ROI, cmap='gray')
                #     plt.title('region_label_ROI')

                    # plt.text(5, 20, 'inside mean: ' + str(round(inside_mean, 2)),
                    #          fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                #
                # if detailplot_tag:
                #
                #
                #     plt.subplot(3, 6, 8)
                #     plt.imshow(region_around_upper, cmap='gray')
                #     plt.title('region_around_upper')

                    # plt.text(5, 20, 'around upper mean: ' + str(round(around_upper_mean, 2)),
                    #          fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                if detailplot_tag:


                    # plt.subplot(3, 6, 9)
                    # plt.imshow(region_around_lower, cmap='gray')
                    # plt.title('region_around_lower')

                    plt.subplot(3, 6, 10)
                    plt.imshow(dyn_2d_win_skeleton, cmap='gray')
                    plt.title('lesion_ring_skeleton')

                    plt.subplot(3, 6, 11)
                    label_skeleton = get_capsule_region_skeleton(label)
                    plt.imshow(label_skeleton, cmap='gray')
                    plt.title('label_skeleton')

                    plt.show()

                # if detailplot_tag:
                #     plt.text(5, 20, 'around lower mean: ' + str(round(around_lower_mean, 2)),
                #              fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                # if detailplot_tag:
                #     plt.text(5, 40, 'inside mean / around upper mean: ' + str(upper_score),
                #              fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                #
                #     plt.text(5, 60, 'inside mean / around lower mean: ' + str(lower_score),
                #              fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))







            if logging_tag:
                print('region ', region_index, ' done!')




    if Capsule_sum >0:

        score = Capsule_sum / lesion_rim_sum

        # if logging_tag:
        #     print('file ', file, ', layer ', str(i+1), ' has capsule region! capsule score is: ', score)
    else:
        score = 0
    #     if logging_tag:
    #         print('file ', file, ', layer ', str(i+1),  ' has no capsule region! capsule score is: ', 0)
    # if logging_tag:
    #     print('file ', file,  ', layer ', str(i+1), ' done!')


    return score


def label_direction(label):
    # calculate direction of the label
    label_copy = copy.deepcopy(label)
    img = np.uint8(label_copy)

    ret, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # minAreaRect = cv2.minAreaRect(contours[0])
    boundingRect = cv2.boundingRect(contours[0])

    # plt.figure()
    # plt.imshow(label, cmap='gray')
    #
    # plt.gca().add_patch(plt.Rectangle((boundingRect[0], boundingRect[1]), boundingRect[2], boundingRect[3]))
    #
    # plt.show()

    if boundingRect[2] / boundingRect[3] < 1:
        direction = 'V'
    else:
        direction = 'H'

    return direction

def mask_segment(t_label_win, region_num, args):
    centerY, centerX = t_label_win.shape[0], t_label_win.shape[1]

    y, x = np.where(t_label_win > 0)[0], np.where(t_label_win > 0)[1]
    y = np.float32(y-np.int0(centerY/2))
    x = np.float32(x-np.int0(centerX/2))



    n = np.arange(len(x))

    r, theta = cv2.cartToPolar(x, y, angleInDegrees=True)
    # xr, yr = cv2.polarToCart(r, theta, angleInDegrees=1)

    color = ['orange', 'green']
    # plt.figure()
    regions = copy.deepcopy(t_label_win)
    for region in range(0,region_num):
        r_ = []
        theta_ = []
        for t in range(0, len(theta)):
            if theta[t][0] >= (360/region_num)*region and theta[t][0] < (360/region_num)*(region+1):
                r_.append(r[t][0])
                theta_.append(theta[t][0])

        r_ = np.array(r_)
        theta_ = np.array(theta_)

        xr_, yr_ = cv2.polarToCart(r_, theta_, angleInDegrees=1)
        xr_ = xr_.tolist()
        yr_ = yr_.tolist()

        xr_ = np.int0(xr_)+np.int0(centerX/2)
        yr_ = np.int0(yr_)+np.int0(centerY/2)

        xr__ = []
        yr__ = []
        for p in range(0, len(xr_)):
            xr__.append(xr_[p][0])
            yr__.append(yr_[p][0])
        # max_value_x = max(xr__)
        # min_value_x = min(xr__)
        # max_value_y = max(yr__)
        # min_value_y = min(yr__)
        # print(min_value_x, max_value_x, min_value_y, max_value_y)
        regions[yr__,xr__] = region

        # print(xr_, yr_)
        #
        # plt.figure(figsize=(9, 5))
        # plt.subplot(221), plt.title("Cartesian coordinate"), plt.plot(x, y, 'o')
        # for i, txt in enumerate(n):
        #     plt.annotate(txt, (x[i], y[i]))
        #
        # plt.subplot(222), plt.title("Polar coordinate"), plt.plot(r, theta, 'o')
        # for i, txt in enumerate(n):
        #     plt.annotate(txt, (r[i], theta[i]))

        # plt.subplot(223), plt.title("Cartesian coordinate sub")
        # if (region % 2) == 0:
        #     plt.plot(xr_, yr_, 'o', color[0])
        # else:
        #     plt.plot(xr_, yr_, 'o', color[1])
        # for i, txt in enumerate(np.arange(len(xr_))):
        #     plt.annotate(txt, (xr_[i], yr_[i]))

        # plt.subplot(224), plt.title("Polar coordinate"), plt.plot(r_, theta_, 'o')
        # for i, txt in enumerate(np.arange(len(r_))):
        #     plt.annotate(txt, (r_[i], theta_[i]))

    # plt.show()
    # 寻找超出预定义区域数的元素。
    # 对于找到的每个这样的点，尝试通过查看该点周围（上下左右）的元素来调整这个点的区域值，使之等于周围点的最小区域值（但仅限于考虑的相邻点）。
    y, x = np.where(regions > args.region_number - 1)[0], np.where(regions > args.region_number - 1)[1]
    for q in range(0, len(x)):
        min_value = regions[y[q], x[q]]
        if y[q] + 1 < centerY:
            min_value = min(regions[y[q] + 1, x[q]],min_value)
        if y[q] - 1 >= 0:
            min_value = min(regions[y[q] - 1, x[q]], min_value)
        if x[q] - 1 >= 0:
            min_value = min(regions[y[q], x[q] - 1], min_value)
        if x[q] + 1 < centerX:
            min_value = min(regions[y[q], x[q] + 1], min_value)

        regions[y[q], x[q]] = min_value

    y, x = np.where(regions <= (args.region_number/4)-1)[0], np.where(regions <= (args.region_number/4)-1)[1]
    for q in range(0, len(x)):
        # regions[y[q], x[q]] = ((args.region_number/2)-1) - regions[y[q], centerX-x[q]]
        corrected_index = min(centerX - x[q], regions.shape[1] - 1)
        regions[y[q], x[q]] = ((args.region_number / 2) - 1) - regions[y[q], corrected_index]

    y, x = np.where(regions == 14)[0], np.where(regions == 14)[1]
    for q in range(0, len(x)):
        regions[y[q], x[q]] = (args.region_number-1) - regions[centerY-y[q]-1, x[q]]

    if args.region_number == 8:
        y, x = np.where(regions == -1)[0], np.where(regions == -1)[1]
        for q in range(0, len(x)):
            regions[y[q], x[q]] = 7
        y, x = np.where(regions == -4)[0], np.where(regions == -4)[1]
        for q in range(0, len(x)):
            regions[y[q], x[q]] = 7
        y, x = np.where(regions == 4)[0], np.where(regions == 4)[1]
        for q in range(0, len(x)):

            regions[y[q], x[q]] = 8+((args.region_number / 2) - 1) - regions[y[q], centerX - x[q]-1]


    return regions


def remove_capsule_region_along_liver_rim2(capsule_region_thresholded_large, l_2d_win_erosion, dyn_2d_win, t_label_win, args):

    capsule_region_thresholded_large_l = copy.deepcopy(capsule_region_thresholded_large)

    y_l, x_l = np.where(l_2d_win_erosion == 0)[0], np.where(l_2d_win_erosion == 0)[1]

    labels, sort_list, stats, centroids, contours = array_proc.region_selection(capsule_region_thresholded_large)

    num_region = labels.max()

    if logging_tag:
        print('Total region num is: ', num_region)


    for region_index in range(1, num_region + 1):
        label = copy.deepcopy(labels)
        y, x = np.where(labels != region_index)
        label[y, x] = 0
        y, x = np.where(labels == region_index)
        label[y, x] = 1

        # calculate lesion rim length
        dyn_2d_win_skeleton = copy.deepcopy(dyn_2d_win)

        y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
        dyn_2d_win_skeleton[y, x] = 0

        t_label_win_erosion_1 = array_proc.erosion(t_label_win, 1)
        y, x = np.where(t_label_win_erosion_1 == 1)[0], np.where(t_label_win_erosion_1 == 1)[1]

        dyn_2d_win_skeleton[y, x] = 0

        y, x = np.where(dyn_2d_win_skeleton > 0)[0], np.where(dyn_2d_win_skeleton > 0)[1]

        dyn_2d_win_skeleton[y, x] = 1

        exist = (dyn_2d_win_skeleton != 0)
        lesion_rim_sum = exist.sum()
        # End calculate lesion rim length
        # 完整候选包膜区域的长度

        label_l = copy.deepcopy(label)

        label_l_skeleton = get_capsule_region_skeleton(label_l)
        exist = label_l_skeleton != 0
        label_l_sum = exist.sum() # 候选包膜区域的长度

        label_l[y_l, x_l] = 0

        label_l_skeleton = get_capsule_region_skeleton(label_l)
        exist = label_l_skeleton != 0
        label_l_sum_eroded = exist.sum() # 肝脏Erosion后候选包膜区域的长度

        if args.compare_with_candidate_capsule_or_tumor_rim_length == 'compare_with_candidate_capsule':
            compare_item = label_l_sum
        if args.compare_with_candidate_capsule_or_tumor_rim_length == 'compare_with_tumor_rim_length':
            compare_item = lesion_rim_sum

        if label_l_sum_eroded < args.liver_rim_erosion_capsule_length_threshold*compare_item: # Erosion掉的部分占总长5%以上即将其认为是肝脏边缘而非包膜，剔除整个候选包膜区域
            if args.erode_candidate_capsule_totally:
                y,x = np.where(label == 1)[0], np.where(label == 1)[1]
                capsule_region_thresholded_large_l[y,x] = 0  # Erosion掉的部分占总长5%以上即将其认为 靠近肝脏边缘处 是肝脏边缘而非包膜，剔除候选包膜区域在肝脏边缘的部分
            else:
                y_l_0_label_1, x_l_0_label_1 = np.where((l_2d_win_erosion == 0) & (label == 1))[0], np.where((l_2d_win_erosion == 0) & (label == 1))[1]
                capsule_region_thresholded_large_l[y_l_0_label_1, x_l_0_label_1] = 0

    return capsule_region_thresholded_large_l

def remove_capsule_region_along_liver_rim1(capsule_region_thresholded_large, l_2d_win_erosion, length_threshold):

    capsule_region_thresholded_large_l = copy.deepcopy(capsule_region_thresholded_large)

    y_l, x_l = np.where(l_2d_win_erosion == 0)[0], np.where(l_2d_win_erosion == 0)[1]

    labels, sort_list, stats, centroids, contours = array_proc.region_selection(capsule_region_thresholded_large)

    num_region = labels.max()

    if logging_tag:
        print('Total region num is: ', num_region)


    for region_index in range(1, num_region + 1):
        label = copy.deepcopy(labels)
        y, x = np.where(labels != region_index)
        label[y, x] = 0
        y, x = np.where(labels == region_index)
        label[y, x] = 1

        exist = (label != 0)
        label_sum = exist.sum()

        label_l = copy.deepcopy(label)
        label_l[y_l, x_l] = 0

        exist = (label_l != 0)
        label_l_sum = exist.sum()

        if label_l_sum < length_threshold*label_sum:
            y,x = np.where(label == 1)[0], np.where(label == 1)[1]
            capsule_region_thresholded_large_l[y,x] = 0

    return capsule_region_thresholded_large_l

def remove_capsule_region_within_tumor(capsule_region_thresholded_large, l_2d_win_erosion, length_threshold):

    capsule_region_thresholded_large_l = copy.deepcopy(capsule_region_thresholded_large)

    y_l, x_l = np.where(l_2d_win_erosion == 0)[0], np.where(l_2d_win_erosion == 0)[1]

    labels, sort_list, stats, centroids, contours = array_proc.region_selection(capsule_region_thresholded_large)

    num_region = labels.max()

    if logging_tag:
        print('Total region num is: ', num_region)


    for region_index in range(1, num_region + 1):
        label = copy.deepcopy(labels)
        y, x = np.where(labels != region_index)
        label[y, x] = 0
        y, x = np.where(labels == region_index)
        label[y, x] = 1

        exist = (label != 0)
        label_sum = exist.sum()

        label_l = copy.deepcopy(label)
        label_l[y_l, x_l] = 0

        exist = (label_l != 0)
        label_l_sum = exist.sum()

        if label_l_sum >= length_threshold*label_sum:
            y,x = np.where(label == 1)[0], np.where(label == 1)[1]
            capsule_region_thresholded_large_l[y,x] = 0

    return capsule_region_thresholded_large_l


def get_ellipse_mask(capsule_region_thresholded_large, radius, t_label_win):
    capsule_region_thresholded_large_copy = copy.deepcopy(capsule_region_thresholded_large)
    capsule_region_thresholded_large_copy = array_proc.dilate_size(capsule_region_thresholded_large, np.int0(0.5*radius)+1)
    y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
    capsule_region_thresholded_large_copy[y, x] = 0

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(capsule_region_thresholded_large, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(capsule_region_thresholded_large_copy, cmap='gray')
    # plt.show()

    ret, binary = cv2.threshold(capsule_region_thresholded_large_copy, 0, 1, cv2.THRESH_BINARY)



    contours, hierarchy = cv2.findContours(binary,
                                                    cv2.RETR_LIST,
                                                    cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = contours[0]
        point_num = len(contour)
        # print(point_num)
        if  point_num > 5:
            ellipse_mask = cv2.fitEllipse(contours[0])


            return ellipse_mask
        else:
            return 0

    else:
        return 0



### ------------------------- End capsule region double confirm ---------------- ############

def ID_write_head(xlspath, header):
    fout1 = open(xlspath, 'w')
    fout1.write(header + '\n')
    fout1.close()



def ID_write(xlspath, ID_status):
    fout1 = open(xlspath, mode='a')
    fout1.write("".join(ID_status) + '\n')
    fout1.close()

outfile = 'xx.xls'
header = "".join(['ID\tlayer\tphase\tregion index\tcapsule region mean\taround upper mean\taround lower mean\ttumor rim middle mean\ttumor rim inside mean\ttumor rim outside mean'])
ID_write_head(outfile, header)


def separate_long_region(input_array):
    # 获取数组的尺寸
    rows, cols = input_array.shape

    # 计算每个区域的大小
    row_split = rows // 2
    col_split = cols // 2

    # 创建一个同样大小的标记数组
    marked_array = np.zeros_like(input_array)

    # 填充每个区域的标记
    # 左上区域标记为0
    marked_array[:row_split, :col_split] = 1
    # 右上区域标记为1
    marked_array[:row_split, col_split:] = 2
    # 右下区域标记为2
    marked_array[row_split:, col_split:] = 3
    # 左下区域标记为3
    marked_array[row_split:, :col_split] = 4

    y, x = np.where(input_array == 0)[0], np.where(input_array == 0)[1]
    marked_array[y, x] = 0

    return marked_array

def capsule_double_confirm0(capsule_region_thresholded_large, dyn_2d_win, t_label_win, t_label_win_, l_2d_win, l_2d_win_erosion, dyn_2d_win_l_t_removed, logging_tag, plotfigure_tag, id, layer, phase, statisticType, args, best_roi_mask):
    dyn_2d_win_copy = copy.deepcopy(dyn_2d_win)
    dyn_2d_win_enhanced = capsule_region_thresholded_large
    lesion_around_contour = display.add_contour_to_img(dyn_2d_win_copy, dyn_2d_win_enhanced, 1,
                                                       (0, 0, 255), 1)

    labels, sort_list, stats, centroids, contours = array_proc.region_selection(capsule_region_thresholded_large)

    num_region = labels.max()
    max_index = num_region
    seperated_index = []

    if args.separate_long_region:
        # 判断每一个联通区域的骨架长度是否大于肿瘤周长的1/4
        # 如果大于1/4，进行拆分；拆分成2X2区域里的4小段。
        _, lesion_rim_sum_ = array_proc.tumor_rim_length(t_label_win)


        for region_index in range(1, num_region + 1):
            # 构建只包含一个candidate capsule region的mask
            label_ = copy.deepcopy(labels)
            y, x = np.where(labels != region_index)
            label_[y, x] = 0
            y, x = np.where(labels == region_index)
            label_[y, x] = 1
            region_length_, _ = array_proc.capsule_region_length(label_)  # 计算当前确定是true capsule region的skeleton length

            if region_length_ > args.long_region_length_k*lesion_rim_sum_:
                seperated_index.append(region_index)
                marked_array = separate_long_region(label_)
                for marker in range(1,5):
                    y_, x_ = np.where(marked_array == marker)[0], np.where(marked_array == marker)[1]
                    if len(x_) > 0:
                        max_index += 1
                        labels[y_, x_] = max_index



    if logging_tag:
        print('Total region num is: ', num_region)

    capsule_tag = []
    true_regions = []

    True_Capsule = copy.deepcopy(t_label_win)
    y,x = np.where(t_label_win==1)[0], np.where(t_label_win==1)[1]
    True_Capsule[y,x] = 0 # 初始化包含所有true capsule regions的mask

    if True:
        # calculate mean intensity of tumor rim
        # 1) middle
        dyn_2d_win_ring = copy.deepcopy(dyn_2d_win_copy)

        y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
        dyn_2d_win_ring[y, x] = 0

        t_label_win_erosion_1 = array_proc.erosion(t_label_win, 1)
        y, x = np.where(t_label_win_erosion_1 == 1)[0], np.where(t_label_win_erosion_1 == 1)[1]

        dyn_2d_win_ring[y, x] = 0

        ring = display.add_contour_to_img(dyn_2d_win_copy, dyn_2d_win_ring, 1, (0, 0, 0), 1)

        exist = (dyn_2d_win_ring != 0)
        tumor_ring_mean_middle = dyn_2d_win_ring.sum() / exist.sum()

        # 2) inside

        dyn_2d_win_ring = copy.deepcopy(dyn_2d_win_copy)

        y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
        dyn_2d_win_ring[y, x] = 0

        t_label_win_erosion_2 = array_proc.erosion(t_label_win, 3)
        y, x = np.where(t_label_win_erosion_2 == 1)[0], np.where(t_label_win_erosion_2 == 1)[1]

        dyn_2d_win_ring[y, x] = 0

        dyn_2d_win_ring = array_proc.erosion(dyn_2d_win_ring, 1)

        ring = display.add_contour_to_img(ring, dyn_2d_win_ring, 0, (0, 0, 255), 1)

        exist = (dyn_2d_win_ring != 0)
        tumor_ring_mean_inside = dyn_2d_win_ring.sum() / exist.sum()

        # 3) outside
        dyn_2d_win_ring = copy.deepcopy(dyn_2d_win_copy)
        t_label_win_dilate = array_proc.dilate_size(t_label_win, 1)
        y,x = np.where(t_label_win_dilate==0)[0], np.where(t_label_win_dilate==0)[1]
        dyn_2d_win_ring[y,x] = 0

        y, x = np.where(t_label_win > 0)[0], np.where(t_label_win > 0)[1]
        dyn_2d_win_ring[y, x] = 0

        ring = display.add_contour_to_img(ring, dyn_2d_win_ring, 0, (255, 0, 0), 1)

        exist = (dyn_2d_win_ring != 0)
        tumor_ring_mean_outside = dyn_2d_win_ring.sum() / exist.sum()


    capsule_sum = 0
    # for region_index in range(1, num_region + 1):
    for region_index in range(1, max_index + 1):
        if region_index not in seperated_index:
            if detailplot_tag:
                plt.figure()
                plt.subplot(3, 6, 1)
                plt.imshow(dyn_2d_win_copy, cmap='gray')
                plt.title('ROI')

                plt.subplot(3, 6, 2)
                plt.imshow(lesion_around_contour, cmap='gray')
                plt.title('ROI with label')
            if logging_tag:
                print('For region: ', region_index)
            # 构建只包含一个candidate capsule region的mask
            label = copy.deepcopy(labels)
            y, x = np.where(labels != region_index)
            label[y, x] = 0
            y, x = np.where(labels == region_index)
            label[y, x] = 1
            # End 构建只包含一个candidate capsule region的mask
            region_label = display.add_contour_to_img(dyn_2d_win_copy, label, 1,
                                                      (0, 255, 0), 1)
            if detailplot_tag:
                plt.subplot(3, 6, 3)
                plt.imshow(region_label, cmap='gray')
                plt.title('region_label')

            if args.capsule_upper_lower_liver_eroded:
                l_2d_win_erosion_for_capsule = array_proc.erosion(l_2d_win, args.liver_rim_erosion_size_for_capsule)
                y_l, x_l = np.where(l_2d_win_erosion_for_capsule == 0)[0], np.where(l_2d_win_erosion_for_capsule == 0)[1]
                label[y_l, x_l] = 0

            if np.max(np.max(label)) > 0:
                if args.dilate_disk:
                    if dilation_kernel == 'cv2':
                        label_dilate = array_proc.dilate(label, args.upper_lower_around_size)
                    if dilation_kernel == 'morphology':
                        label_dilate = array_proc.dilate_morphologyKernel_cv2Dilation(label, args.upper_lower_around_size)
                else:
                    kernel = np.array([[0, 1, 0],
                                       [1, 1, 1],
                                       [0, 1, 0]], dtype=np.uint8)  # 创建3x3的方形内核，只有边缘是1
                    label_dilate = morphology.dilation(label, kernel)


                y, x = np.where(label == region_index)[0], np.where(label == region_index)[1]
                label_dilate[y, x] = 0

                if args.capsule_upper_lower_liver_eroded:
                    l_2d_win_erosion_for_capsule = array_proc.erosion(l_2d_win, args.liver_rim_erosion_size_for_capsule)
                    y_l, x_l = np.where(l_2d_win_erosion_for_capsule == 0)[0], np.where(l_2d_win_erosion_for_capsule == 0)[1]
                    label_dilate[y_l, x_l] = 0

                    if args.remove_bright_region or args.remove_dark_region:
                        y, x = np.where(dyn_2d_win_l_t_removed == 0)[0], np.where(dyn_2d_win_l_t_removed== 0)[1]
                        label_dilate[y, x] = 0

                if np.max(np.max(label_dilate)) > 0:

                    if detailplot_tag:
                        plt.subplot(3, 6, 4)
                        plt.imshow(label_dilate, cmap='gray')
                        plt.title('label_dilate')

                    # label_skeleton = morphology.skeletonize(label)
                    #
                    # region_label_skeleton = display.add_contour_to_img(region_label, label_skeleton, 0,
                    #                                           (0, 0, 255), 1)
                    if detailplot_tag:
                        plt.subplot(3, 6, 5)
                        plt.imshow(region_label, cmap='gray')
                        plt.title('region_label')

                    # Get upper and lower edge of the capsule region
                    y, x = np.where(label == 1)[0], np.where(label == 1)[1]

                    label_left_x = min(x)
                    label_right_x = max(x)

                    X = []
                    Y_max = []
                    Y_min = []
                    for x_value in range(label_left_x, label_right_x + 1):
                        X.append(x_value)
                        x_index = np.argwhere(x == x_value) # 返回非0的数组元组的索引，括号里是要索引数组的条件。

                        y_max = min(y)
                        y_min = max(y)
                        for index in range(0, len(x_index)):
                            y_index = x_index[index][0]
                            y_value = y[y_index]
                            if y_value > y_max:
                                y_max = y_value
                            if y_value < y_min:
                                y_min = y_value

                        Y_max.append(y_max)
                        Y_min.append(y_min)

                    if detailplot_tag:
                        for col in range(0, len(X)):
                            plt.scatter(X[col], Y_max[col], color='orange')
                            plt.scatter(X[col], Y_min[col], color='purple')

                    # End Get upper and lower edge of the capsule region

                    ###############################################################################################
                    ##################  Seperate upper and lower part of the capsule around region  ###############
                    # Step 1: 对于每个x，找到上下两个区域的所有点，候选包膜区域，以及2 pixels的Gap
                    upper_y = []
                    upper_x = []
                    lower_y = []
                    lower_x = []

                    all_y = []
                    all_x = []

                    y_label_dilate, x_label_dilate = np.where(label_dilate == 1)[0], \
                                                     np.where(label_dilate == 1)[1]

                    for x_value_index in range(0, len(X)):
                        x_value = X[x_value_index]
                        y_max = Y_max[x_value_index]
                        y_min = Y_min[x_value_index]

                        x_index = np.argwhere(x_label_dilate == x_value)

                        for index in range(0, len(x_index)):
                            y_index = x_index[index][0]
                            y_value = y_label_dilate[y_index]

                            if y_value > y_max + args.inside_around_gap:
                                lower_x.append(x_value)
                                lower_y.append(y_value)

                                all_x.append(x_value)
                                all_y.append(y_value)

                            if y_value < y_min - args.inside_around_gap:
                                upper_x.append(x_value)
                                upper_y.append(y_value)

                                all_x.append(x_value)
                                all_y.append(y_value)
                    # End Step 1: 对于每个x，找到上下两个区域的所有点，候选包膜区域，以及2 pixels的Gap


                    # Step 2: 构建上下两个区域对应的mask
                    mask = np.zeros((labels.shape[0], labels.shape[1]))
                    mask[all_y, all_x] = 1
                    y, x = np.where(mask == 0)[0], np.where(mask == 0)[1]

                    label_dilate_upper = copy.deepcopy(label_dilate)
                    label_dilate_upper[y, x] = 0
                    label_dilate_upper[lower_y, lower_x] = 0

                    label_dilate_lower = copy.deepcopy(label_dilate)
                    label_dilate_lower[y, x] = 0
                    label_dilate_lower[upper_y, upper_x] = 0

                    # End Step 2: 构建上下两个区域对应的mask


                    if args.normalize_tag:
                        cv2.normalize(dyn_2d_win_copy, dyn_2d_win_copy, 0, 255, cv2.NORM_MINMAX)
                    y, x = np.where(label == 0)[0], np.where(label == 0)[1]
                    region_label_ROI = copy.deepcopy(dyn_2d_win_copy)
                    region_label_ROI[y, x] = 0  # 原图（0-255）只保留某个candidate capsule region

                    y, x = np.where(label_dilate_upper == 0)[0], np.where(label_dilate_upper == 0)[1]
                    region_around_upper = copy.deepcopy(dyn_2d_win_copy)
                    region_around_upper[y, x] = 0  # 原图（0-255）只保留candidate capsule region上面那个区域

                    y, x = np.where(label_dilate_lower == 0)[0], np.where(label_dilate_lower == 0)[1]
                    region_around_lower = copy.deepcopy(dyn_2d_win_copy)
                    region_around_lower[y, x] = 0  # 原图（0-255）只保留candidate capsule region下面那个区域

                    if (auto_segmentation and (not use_manual_parameters)):
                        # 当包膜位于肝脏边缘时，如果候选包膜两侧的区域太小，或者直接没有肝脏区域，则认为该候选包膜区域不是真正的包膜。
                        exist = (label != 0)
                        label_sum = exist.sum()  # 完整候选包膜区域的长度（用像素点总和近似）

                        exist = (label_dilate_upper != 0)
                        label_upper_sum = exist.sum()  # 候选包膜一侧区域的长度（用像素点总和近似）

                        exist = (label_dilate_lower != 0)
                        label_lower_sum = exist.sum()  # 候选包膜另外一侧区域的长度（用像素点总和近似）

                        # 判断 region_around_upper 和 region_around_lower哪个在肿瘤内侧，哪个在肿瘤外侧 (tumor mask 不准确时，可能影响判断的准确性)
                        upper_overlap = copy.deepcopy(t_label_win)
                        y, x = np.where(region_around_upper == 0)[0], np.where(region_around_upper == 0)[1]
                        upper_overlap[y,x] = 0
                        exist = (upper_overlap != 0)
                        upper_overlap_sum = exist.sum()

                        lower_overlap = copy.deepcopy(t_label_win)
                        y, x = np.where(region_around_lower == 0)[0], np.where(region_around_lower == 0)[1]
                        lower_overlap[y,x] = 0
                        exist = (lower_overlap != 0)
                        lower_overlap_sum = exist.sum()


                        if upper_overlap_sum >= lower_overlap_sum: # region_around_upper在肿瘤内侧
                            if label_lower_sum < args.capsule_upper_lower_area_ratio*label_sum:
                                if best_roi_mask is not None:
                                    y,x = np.where(best_roi_mask == 0)[0], np.where(best_roi_mask == 0)[1]
                                    region_around_lower = copy.deepcopy(dyn_2d_win)
                                    region_around_lower[y,x] = 0
                                    exist = (region_around_lower != 0)
                                    label_lower_sum = exist.sum()  # 候选包膜一侧区域的长度（用像素点总和近似）
                        else:
                            if label_upper_sum < args.capsule_upper_lower_area_ratio*label_sum:
                                if best_roi_mask is not None:
                                    y, x = np.where(best_roi_mask == 0)[0], np.where(best_roi_mask == 0)[1]
                                    region_around_upper = copy.deepcopy(dyn_2d_win)
                                    region_around_upper[y, x] = 0
                                    exist = (region_around_upper != 0)
                                    label_upper_sum = exist.sum()  # 候选包膜一侧区域的长度（用像素点总和近似）

                        condition_ = label_upper_sum >= args.capsule_upper_lower_area_ratio*label_sum and label_lower_sum >= args.capsule_upper_lower_area_ratio*label_sum
                        # condition_ = True # For comparison method
                    else:
                        condition_ = True

                    if condition_:
                        #############################  End Seperate upper and lower part of the capsule around region ######################
                        #####################################################################################################################


                        if detailplot_tag:
                            plt.subplot(3, 6, 7)
                            plt.imshow(region_label_ROI, cmap='gray')
                            plt.title('region_label_ROI')

                        if statisticType == 'mean':
                            exist = (region_label_ROI != 0)
                            inside_mean = region_label_ROI.sum() / exist.sum() # mean intensity of candidate capsule region
                        if statisticType == 'percentile':
                            region_label_ROI_positive = region_label_ROI.ravel()[np.flatnonzero(region_label_ROI)]
                            inside_mean = np.percentile(region_label_ROI_positive,75)

                        if detailplot_tag:
                            plt.text(5, 20, 'inside mean: ' + str(round(inside_mean, 2)),
                                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                            plt.subplot(3, 6, 8)
                            plt.imshow(region_around_upper, cmap='gray')
                            plt.title('region_around_upper')

                        if statisticType == 'mean' or statisticType == 'percentile':
                            exist = (region_around_upper != 0)
                            around_upper_mean = region_around_upper.sum() / exist.sum() # mean intensity of upper region
                        # if statisticType == 'percentile':
                        #     region_around_upper_positive = region_around_upper.ravel()[np.flatnonzero(region_around_upper)]
                        #     around_upper_mean = np.percentile(region_around_upper_positive,80)

                        if detailplot_tag:
                            plt.text(5, 20, 'around upper mean: ' + str(round(around_upper_mean, 2)),
                                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                            plt.subplot(3, 6, 9)
                            plt.imshow(region_around_lower, cmap='gray')
                            plt.title('region_around_lower')

                        if statisticType == 'mean' or statisticType == 'percentile':
                            exist = (region_around_lower != 0)
                            around_lower_mean = region_around_lower.sum() / exist.sum() # mean intensity of lower region
                        # if statisticType == 'percentile':
                        #     region_around_lower_positive = region_around_lower.ravel()[np.flatnonzero(region_around_lower)]
                        #     around_lower_mean = np.percentile(region_around_lower_positive,80)

                        if detailplot_tag:
                            plt.text(5, 20, 'around lower mean: ' + str(round(around_lower_mean, 2)),
                                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))
                        # 计算2个比值
                        upper_score = round(inside_mean / around_upper_mean, 2)
                        lower_score = round(inside_mean / around_lower_mean, 2)

                        # 计算2个插值，目前没有使用
                        upper_score_ = round(inside_mean - around_upper_mean, 2)
                        lower_score_ = round(inside_mean - around_lower_mean, 2)

                        # export capsule intensity
                        ID_write(outfile,
                                 [id, '\t', str(layer + 1), '\t', phase, '\t', str(region_index), '\t', str(inside_mean), '\t',
                                  str(around_upper_mean), '\t', str(around_lower_mean), '\t',
                                  str(tumor_ring_mean_middle), '\t', str(tumor_ring_mean_inside), '\t', str(tumor_ring_mean_outside)])


                        if detailplot_tag:
                            plt.text(5, 40, 'inside mean / around upper mean: ' + str(upper_score) + ', ' + str(upper_score_),
                                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                            plt.text(5, 60, 'inside mean / around lower mean: ' + str(lower_score) + ', ' + str(lower_score_),
                                     fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

                        # if ((upper_score >= score_threshold) and (lower_score >= score_threshold)) and (max(upper_score, lower_score) < 1.4):
                        # if ((max(upper_score_, lower_score_) >= 40) and (min(upper_score_, lower_score_) >= 15)) and (
                        #             max(upper_score, lower_score) < 1.4) and (upper_score_<30 or lower_score_ <30):
                        # if ((max(upper_score_, lower_score_) >= 40) and (min(upper_score_, lower_score_) >= 15)) and (
                        #             max(upper_score, lower_score) < 1.4):
                        if auto_segmentation and (not use_manual_parameters):

                            if args.data_site == 'ZhongShan':
                                condition = (upper_score >= args.score_threshold) and (lower_score >= args.score_threshold)
                                # condition = (max(upper_score_, lower_score_) >= deviation_threshold_max) and (min(upper_score_, lower_score_) >= deviation_threshold_min)
                            if args.data_site == 'SuZhou' or args.data_site == 'PHC':
                                # condition = (max(upper_score_, lower_score_) >= deviation_threshold_max) and (min(upper_score_, lower_score_) >= deviation_threshold_min)

                                    condition = (upper_score > args.score_threshold) and (lower_score > args.score_threshold)
                        else:
                            if Capsule_Updated:
                                condition = (upper_score > args.score_threshold) and (lower_score > args.score_threshold)
                            else:
                                condition = (upper_score >= args.score_threshold) and (lower_score >= args.score_threshold)

                        # condition = True # 用于消融实验
                        if condition:
                            if logging_tag:
                                print('region ', region_index, ' is capsule region!')
                            capsule_tag.append(min(upper_score, lower_score))
                            true_regions.append(region_label_ROI)

                            region_copy = copy.deepcopy(region_label_ROI)
                            y,x = np.where(region_label_ROI>0)[0], np.where(region_label_ROI>0)[1]
                            region_copy[y,x] = 1 # 当前确定是true capsule region的binary mask

                            region_length, region_skeleton = array_proc.capsule_region_length(region_copy) # 计算当前确定是true capsule region的skeleton length

                            if not args.weighted_capsule_length:
                                capsule_sum = capsule_sum + region_length
                            else:
                                capsule_sum = capsule_sum + region_length*max(upper_score, lower_score)


                            y,x = np.where(region_label_ROI>0)[0], np.where(region_label_ROI>0)[1]
                            True_Capsule[y,x] = 1 # 更新包含所有true capsule regions的mask



                        else:
                            capsule_tag.append(0)

                    if plotfigure_tag:
                        plt.subplot(3, 6, 13)
                        plt.imshow(region_label_ROI, cmap='gray')
                        plt.title('region_label_ROI')

                        plt.subplot(3, 6, 14)
                        plt.imshow(region_around_upper, cmap='gray')
                        plt.title('region_around_upper')

                        plt.subplot(3, 6, 15)
                        plt.imshow(region_around_lower, cmap='gray')
                        plt.title('region_around_lower')

                        region_label_upper = display.add_contour_to_img(region_label, region_around_upper, 0,
                                                                  (0, 255, 255), 1) # Cyan
                        region_label_upper_lower = display.add_contour_to_img(region_label_upper, region_around_lower, 0,
                                                                        (255, 255, 0), 1)  # Yellow

                        plt.subplot(3, 6, 16)
                        plt.imshow(region_label_upper_lower, cmap='gray')
                        plt.title('region_around_upper_lower')

                        plt.show()

                    if logging_tag:
                        print('region ', region_index, ' done!')

    if plotfigure_tag:

        if len(true_regions) > 0:
            # true_capsule_region = display.add_contour_to_img(dyn_2d_win_copy, true_regions[0], 1,
            #                                                  (0, 0, 255), 1)
            true_capsule_region = display.add_contour_to_img(dyn_2d_win, true_regions[0], 1,
                                                             (0, 255, 0), 1)
            if len(true_regions) > 1:
                for true_capsule_region_index in range(1, len(true_regions)):
                    true_capsule_region = display.add_contour_to_img(true_capsule_region, true_regions[true_capsule_region_index], 0,
                                                                    (0, 255, 0), 1)




    if capsule_sum > 0:
        # calculate lesion rim length
        dyn_2d_win_skeleton, lesion_rim_sum = array_proc.tumor_rim_length(t_label_win)

        if plotfigure_tag:
            # plt.figure()
            # plt.imshow(dyn_2d_win_skeleton, cmap='gray')
            # plt.title('lesion_skeleton')
            #
            # plt.show()
            if len(true_regions) > 0:
                plt.figure()
                plt.imshow(true_capsule_region, cmap='gray')
                plt.title('true capsule regions1')

                plt.show()


        # End calculate lesion rim length

        score = capsule_sum / lesion_rim_sum

        # if logging_tag:
        #     print('file ', file, ', layer ', str(i + 1), ' has capsule region! capsule score is: ', score)
    else:
        score = 0
        # if logging_tag:
        #     print('layer ', str(i + 1), ' has no capsule region! capsule score is: ', 0)
    # if logging_tag:
    #     print('layer ', str(i + 1), ' done!')

    return score


### ------------------------- End capsule region double confirm ---------------- ############



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

def roi_all_larger_than_zero0(arr, roi_size=(20, 20)):
    # 获取数组的维度
    rows, cols = arr.shape

    valid_position = np.zeros((rows - roi_size[0] + 1, cols - roi_size[1] + 1), dtype=bool)

    # 遍历可能的起点位置
    for i in range(rows - roi_size[0] + 1):
        for j in range(cols - roi_size[1] + 1):
            # 获取当前ROI
            current_roi = arr[i:i + roi_size[0], j:j + roi_size[1]]

            # 检查ROI中是否所有值都大于0
            if np.all(current_roi > 0):
                valid_position[i,j] = 1

    return valid_position


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


from scipy.ndimage import generic_filter

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


def compare_candidate_capsule_with_best_roi(dyn_2d_win_l_t, capsule_region_thresholded_large, best_roi_mask, bright_k):
    # 计算best_roi_mask对应的原图的平均信号强度
    dyn_2d_win_l_t_ROI = copy.deepcopy(dyn_2d_win_l_t)
    y_b, x_b = np.where(best_roi_mask == 0)[0], np.where(best_roi_mask == 0)[1]
    dyn_2d_win_l_t_ROI[y_b, x_b] = 0
    dyn_2d_win_l_t_ROI_mean = np.mean(dyn_2d_win_l_t_ROI[dyn_2d_win_l_t_ROI != 0])

    # print('ROI mean: ', dyn_2d_win_l_t_ROI_mean)

    capsule_region_thresholded_large_remove_bright = copy.deepcopy(capsule_region_thresholded_large)

    labels, sort_list, stats, centroids, contours = array_proc.region_selection(capsule_region_thresholded_large)

    num_region = labels.max()

    for region_index in range(1, num_region + 1):
        # 构建只包含一个candidate capsule region的原图，计算该label的平均信号强度
        dyn_2d_win_l_t_label = copy.deepcopy(dyn_2d_win_l_t)
        y, x = np.where(labels != region_index)
        dyn_2d_win_l_t_label[y,x] = 0

        if np.max(dyn_2d_win_l_t_label) > 0:
            dyn_2d_win_l_t_label_mean = np.mean(dyn_2d_win_l_t_label[dyn_2d_win_l_t_label != 0])

            # flattened_nonzero = dyn_2d_win_l_t_label[dyn_2d_win_l_t_label != 0].flatten()
            # dyn_2d_win_l_t_label_95_percentile = np.percentile(flattened_nonzero, 95)

            # print('label mean / ROI mean: ', dyn_2d_win_l_t_label_mean/dyn_2d_win_l_t_ROI_mean)

            if dyn_2d_win_l_t_label_mean > dyn_2d_win_l_t_ROI_mean*bright_k:
            # if dyn_2d_win_l_t_label_95_percentile > dyn_2d_win_l_t_ROI_mean * bright_k:
                y, x = np.where(labels == region_index)
                capsule_region_thresholded_large_remove_bright[y,x] = 0

    return capsule_region_thresholded_large_remove_bright, dyn_2d_win_l_t_ROI_mean



def Capsule2(dyn_2d_win, t_label_win, l_2d_win, liver_without_vessel_win, liver_without_vessel_win_remove_large,
                       padding_tag, frangi_thresholded_tag, radius, inside_threshold, outside_threshold,
                       dilate_k, frangi_threshold, plotfigure_tag, phase, vessel_removal, remove_tumor_dark, liver_rim_erosion, id, layer, args):

    # -------------------- Remove Tumor Dark --------------------------------------------------------
    if remove_tumor_dark:
        adaptive_threshold_mask_tumor = 15 #35
        # segment tumor to remove very dark region
        # Get liver region
        tumor_region = copy.deepcopy(dyn_2d_win)
        y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
        tumor_region[y, x] = 0

        dyn_2d_normalized = copy.deepcopy(tumor_region)
        cv2.normalize(tumor_region, dyn_2d_normalized, 0, 255, cv2.NORM_MINMAX)

        # threshTwoPeaks(dyn_2d_normalized.astype(np.uint8))
        kernel_size = min(adaptive_threshold_mask_tumor, 2 * np.int0(0.5 * radius) + 1)
        if kernel_size == 1:
            kernel_size = 3

        result = cv2.adaptiveThreshold(dyn_2d_normalized.astype(np.uint8), 255,
                                              cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                              kernel_size,
                                              0)

        # ret, result = cv2.threshold(dyn_2d_normalized.astype(np.uint8), 160, 255, cv2.THRESH_BINARY)  # 160
        # ret, result = cv2.threshold(dyn_2d_normalized.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # ret, result = cv2.threshold(dyn_2d_normalized.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

        result_large = copy.deepcopy(result)

        result_large = array_proc.dilate_size(result_large, 1)
        result_large = array_proc.erosion(result_large, 1)

        # result_large = array_proc.remove_small_t(result_large, 20)
        #
        # exist = (result_large != 0)
        # large_sum = exist.sum()

        # result = array_proc.hole_fill(result)
    # ----------------------- End Remove Tumor Dark -----------------------------------------------------





    dyn_2d_win_lesion_around = get_lesion_around(dyn_2d_win, t_label_win, l_2d_win, radius)


    # Calculate Gradient
    # gradient = img_gradient(dyn_2d_win, t_label_win, radius)
    #
    # thresh, ret = cv2.threshold(gradient, 100, 255, cv2.THRESH_BINARY)

    if padding_tag:
        dyn_2d_win_lesion_around_padding, y_before_padding, x_before_padding = paddingHV(dyn_2d_win_lesion_around)
        dyn_2d_win_double = cv2.normalize(dyn_2d_win_lesion_around_padding.astype('double'), None, 0.0, 1.0,
                                          cv2.NORM_MINMAX)  # Convert to normalized floating point
    else:
        dyn_2d_win_double = cv2.normalize(dyn_2d_win_lesion_around.astype('double'), None, 0.0, 1.0,
                                          cv2.NORM_MINMAX)  # Convert to normalized floating point

    # dyn_2d_win_enhanced, angles, mu1, mu2, v1x, v1y, v2x, v2y, S2 = FrangiFilter2D.FrangiFilter2D(dyn_2d_win_double, args.frangiBetaTwo)
    dyn_2d_win_enhanced, angles = FrangiFilter2D.FrangiFilter2D(dyn_2d_win_double, args.frangiBetaTwo)

    y_close_dis_inside, x_close_dis_inside, y_close_dis_outside, x_close_dis_outside = \
        tumor_distance_transform(t_label_win, inside_threshold, radius, dilate_k, outside_threshold,args)

    capsule_mask = copy.deepcopy(dyn_2d_win)
    capsule_mask[y_close_dis_inside, x_close_dis_inside] = 0
    capsule_mask[y_close_dis_outside, x_close_dis_outside] = 0
    y,x = np.where(capsule_mask > 0)[0],np.where(capsule_mask > 0)[1]
    capsule_mask[y,x] = 1


    if frangi_thresholded_tag:
        if args.thresholdingFirstThenRingRegion:
            if args.data_site == 'ZheYi':
                frangi_thresholded, frangi_thresholded_large = frangi_thresholding_false_capsule_based(dyn_2d_win_enhanced, t_label_win, l_2d_win, radius,args)
            if args.data_site == 'ZhongShan' or args.data_site == 'SuZhou' or args.data_site == 'PHC':
                frangi_thresholded, frangi_thresholded_large = frangi_thresholding(dyn_2d_win_enhanced, frangi_threshold,
                                                                               radius, args)
                # frangi_thresholded, frangi_thresholded_large = frangi_thresholding_false_capsule_based(dyn_2d_win_enhanced,
                #                                                                            t_label_win,
                #                                                                            l_2d_win, radius)

            if padding_tag:
                capsule_region = get_capsule_region_padding(frangi_thresholded_large, y_close_dis_inside, x_close_dis_inside, y_close_dis_outside,
                                           x_close_dis_outside, y_before_padding, x_before_padding)
            else:
                capsule_region = get_capsule_region_no_padding(frangi_thresholded_large, y_close_dis_inside, x_close_dis_inside, y_close_dis_outside,
                                              x_close_dis_outside)
        else:
            if padding_tag:
                capsule_region_withoutThresholding = get_capsule_region_padding(dyn_2d_win_enhanced, y_close_dis_inside, x_close_dis_inside, y_close_dis_outside,
                                           x_close_dis_outside, y_before_padding, x_before_padding)
            else:
                capsule_region_withoutThresholding = get_capsule_region_no_padding(dyn_2d_win_enhanced, y_close_dis_inside, x_close_dis_inside, y_close_dis_outside,
                                              x_close_dis_outside)

            if args.data_site == 'ZheYi':
                _, capsule_region = frangi_thresholding_false_capsule_based(capsule_region_withoutThresholding, t_label_win, l_2d_win, radius,args)
            if args.data_site == 'ZhongShan' or args.data_site == 'SuZhou' or args.data_site == 'PHC':
                _, capsule_region = frangi_thresholding(capsule_region_withoutThresholding, frangi_threshold, radius, args)

    else:
        if padding_tag:
            capsule_region = get_capsule_region_padding(dyn_2d_win_enhanced, y_close_dis_inside, x_close_dis_inside, y_close_dis_outside,
                                       x_close_dis_outside, y_before_padding, x_before_padding)
        else:
            capsule_region = get_capsule_region_no_padding(dyn_2d_win_enhanced, y_close_dis_inside, x_close_dis_inside, y_close_dis_outside,
                                          x_close_dis_outside)


    # Adaptive threshold for Capsule Region
    if args.data_site == 'ZheYi':
        capsule_region_thresholded, capsule_region_thresholded_large = capsule_region_thresholding_OTSU(capsule_region, radius)

    if args.data_site == 'ZhongShan' or args.data_site == 'SuZhou' or args.data_site == 'PHC':
        capsule_region_thresholded, capsule_region_thresholded_large = capsule_region_thresholding(capsule_region, radius)
        # capsule_region_thresholded, capsule_region_thresholded_large = capsule_region_thresholding_OTSU(capsule_region,
        #                                                                                                 radius)

    # # calculate capsule score based on skeleton
    # if padding_tag:
    #     dyn_2d_win_skeleton = lesion_ring_skeleton(dyn_2d_win_lesion_around_padding, t_label_win)
    # else:
    #     dyn_2d_win_skeleton = lesion_ring_skeleton(dyn_2d_win_lesion_around, t_label_win)
    #
    # capsule_region_skeleton = get_capsule_region_skeleton(capsule_region_thresholded_large)

    # y_close_liver, x_close_liver = tumor_around_liver(l_2d_win, l_dis_2d_win, dis_threshold)
    # dyn_2d_win_skeleton[y_close_liver, x_close_liver] = 0
    # capsule_region_skeleton[y_close_liver, x_close_liver] = 0

    # ret_dilated = array_proc.dilate(ret, 1)
    # # ret_dilated = copy.deepcopy(ret)
    # y, x = np.where(ret_dilated == 0)[0], np.where(ret_dilated == 0)[1]
    # capsule_region_skeleton_gradient = copy.deepcopy(capsule_region_skeleton)
    # capsule_region_skeleton_gradient[y, x] = 0


    # capsule_score = np.sum(capsule_region_skeleton_gradient.astype(np.int)) / np.sum(dyn_2d_win_skeleton)

    # y, x = np.where(ret == 255)[0], np.where(ret == 255)[1]
    # ret[y, x] = 1
    # ret = get_capsule_region_skeleton(ret)
    # capsule_score = np.sum(ret.astype(np.int)) / np.sum(dyn_2d_win_skeleton)

    # Remove capsule edges inside
    ellipse_mask = copy.deepcopy(capsule_region_thresholded_large)
    y,x = np.where(ellipse_mask == 1)[0], np.where(ellipse_mask == 1)[1]
    ellipse_mask[y,x] = 0

    ellipse = 0
    if args.remove_capsule_within_tumor:
        ellipse = get_ellipse_mask(capsule_region_thresholded_large, radius, t_label_win)

    if isinstance(ellipse, tuple) == True:
        cv2.ellipse(ellipse_mask, ellipse, (255, 0, 0), -1)
        y,x = np.where(ellipse_mask == 255)[0], np.where(ellipse_mask == 255)[1]
        ellipse_mask[y,x] = 1


        # ellipse_mask = array_proc.erosion(ellipse_mask, np.int0(0.4*radius)+1)


        plt.figure()
        plt.subplot(221)
        plt.imshow(ellipse_mask, cmap='gray')

        ellipse_mask_contour = display.add_contour_to_img(capsule_region_thresholded_large, ellipse_mask, 1,
                                                                   (0, 0, 255), 1)

        plt.subplot(222)
        plt.imshow(ellipse_mask_contour, cmap='gray')


        ellipse_mask_erosion = array_proc.erosion(ellipse_mask, np.int0(0.3*radius)+1)
        # ellipse_mask_erosion = array_proc.erosion(ellipse_mask, 5)

        ellipse_mask_erosion_contour = display.add_contour_to_img(capsule_region_thresholded_large, ellipse_mask_erosion, 1,
                                                          (0, 0, 255), 1)


        plt.subplot(223)
        plt.imshow(ellipse_mask_erosion, cmap='gray')
        plt.subplot(224)
        plt.imshow(ellipse_mask_erosion_contour, cmap='gray')

        # plt.show()

        # Remove capsule region inside the tumor
        # y,x = np.where(ellipse_mask_erosion == 1)[0], np.where(ellipse_mask_erosion == 1)[1]
        # capsule_region_thresholded_large[y,x] = 0
        capsule_region_thresholded_large = remove_capsule_region_within_tumor(capsule_region_thresholded_large,
                                                                                         ellipse_mask_erosion, length_threshold=0.4)

    # remove very small regions
    # 1) total pixels < 3
    # capsule_region_thresholded_large = array_proc.remove_small_t(capsule_region_thresholded_large, 3)
    # 2) skeleton length < threshold
    dyn_2d_win_skeleton, lesion_rim_sum = array_proc.tumor_rim_length(t_label_win)
    # capsule_region_thresholded_large = array_proc.remove_small_length(capsule_region_thresholded_large, length = max(4,0.2*lesion_rim_sum))

    y_t, x_t = np.where(t_label_win == 1)[0], np.where(t_label_win == 1)[1]
    l_2d_win_l = copy.deepcopy(l_2d_win)
    l_2d_win_l[y_t, x_t] = 0

    y_l_t, x_l_t = np.where(l_2d_win_l == 0)[0], np.where(l_2d_win_l == 0)[1]
    dyn_2d_win_l_t = copy.deepcopy(dyn_2d_win)
    dyn_2d_win_l_t[y_l_t, x_l_t] = 0

    dyn_2d_win_l_t_removed_dark = copy.deepcopy(dyn_2d_win_l_t)
    dyn_2d_win_l_t_removed = copy.deepcopy(dyn_2d_win_l_t)

    # --------------- Remove very dark region --------------------
    if args.remove_dark_region or args.remove_bright_region:

        # 将二维像素值按照信号强度排序，并将信号值较小的2%置0，去暗区域
        # 将二维数组展平，并去除所有的0值
        flattened_nonzero = dyn_2d_win_l_t[dyn_2d_win_l_t != 0].flatten()


        # 如果去除0后数组为空，则直接返回原数组
        if flattened_nonzero.size == 0:
            dyn_2d_win_l_t_removed_dark = dyn_2d_win_l_t
            dyn_2d_win_l_t_removed = dyn_2d_win_l_t
            best_roi_mask = None
        else:
            if plotfigure_tag:
                plt.figure()
                plt.subplot(2, 4, 1)
                plt.imshow(dyn_2d_win_l_t, cmap='gray')

            if args.remove_dark_region:
                # 计算需要置0的阈值，即去除0值后顶部5%的信号强度
                threshold = np.percentile(flattened_nonzero, args.remove_dark_liver_ratio)


                # 将所有大于或等于阈值的元素置为0
                dyn_2d_win_l_t_removed_dark[dyn_2d_win_l_t_removed_dark <= threshold] = 0

                if plotfigure_tag:
                    plt.subplot(2, 4, 2)
                    plt.imshow(dyn_2d_win_l_t_removed_dark, cmap='gray')

            # --------------- Remove very bright region, mainly blood vessel --------------------
            if args.remove_bright_region:
                if flattened_nonzero.size == 0:
                    dyn_2d_win_l_t_removed = dyn_2d_win_l_t
                else:
                    # 计算需要置0的阈值，即去除0值后顶部5%的信号强度
                    threshold = np.percentile(flattened_nonzero, args.remove_bright_liver_ratio)

                    dyn_2d_win_l_t_removed[dyn_2d_win_l_t_removed >= threshold] = 0


                    if plotfigure_tag:
                        plt.subplot(2, 4, 3)
                        plt.imshow(dyn_2d_win_l_t_removed, cmap='gray')


                    # --------------- 在去除大血管后的肝脏区域选择ROI，用于比较待选包膜区域是否属于高亮区域 ----------------
                    ROI_size = 20  # 选20*20的ROI区域
                    best_roi, best_roi_mask = find_min_cv_roi(dyn_2d_win_l_t_removed, roi_size=(ROI_size, ROI_size))
                    if best_roi is None:
                        best_roi, best_roi_mask = find_min_cv_roi(dyn_2d_win_l_t_removed, roi_size=(15, 15))
                        if best_roi is None:
                            best_roi, best_roi_mask = find_min_cv_roi(dyn_2d_win_l_t_removed, roi_size=(10, 10))
                            if best_roi is None:
                                best_roi, best_roi_mask = find_min_cv_roi(dyn_2d_win_l_t_removed, roi_size=(5, 5))

                    dyn_2d_win_l_t_removed_best_roi_mask = display.add_contour_to_img(dyn_2d_win_l_t_removed, best_roi_mask, 1,
                                                                         (0, 255, 0), 1)

                    if plotfigure_tag:
                        plt.subplot(2, 4, 4)
                        plt.imshow(dyn_2d_win_l_t_removed_best_roi_mask, cmap='gray')

                        plt.subplot(2, 4, 5)
                        plt.imshow(dyn_2d_win, cmap='gray')


                    if plotfigure_tag:
                        capsule_region_thresholded_large_mask = display.add_contour_to_img(dyn_2d_win,
                                                                                                 capsule_region_thresholded_large, 1,
                                                                                           (0, 255, 0), 1)
                        plt.subplot(2, 4, 6)
                        plt.imshow(capsule_region_thresholded_large_mask, cmap='gray')

                    # y, x = np.where(dyn_2d_win_l_t_removed == 0)[0], np.where(dyn_2d_win_l_t_removed == 0)[1]
                    # capsule_region_thresholded_large[y, x] = 0
                    # capsule_region_thresholded_large_mask = display.add_contour_to_img(dyn_2d_win,
                    #                                                                          capsule_region_thresholded_large, 1,
                    #                                                                          (0, 255, 0), 1)
                    # plt.subplot(2, 4, 7)
                    # plt.imshow(capsule_region_thresholded_large_mask, cmap='gray')


                    bright_k = args.remove_bright_candidate_capsule_k

                    if best_roi is not None: # 将每一个待选包膜区域与ROI区域进行比较
                        capsule_region_thresholded_large, dyn_2d_win_l_t_ROI_mean = compare_candidate_capsule_with_best_roi(dyn_2d_win_l_t, capsule_region_thresholded_large, best_roi_mask,
                                                                bright_k)
                    if plotfigure_tag:
                        plt.subplot(2, 4, 7)
                        capsule_region_thresholded_large_mask_removed = display.add_contour_to_img(dyn_2d_win,
                                                                                           capsule_region_thresholded_large, 1,
                                                                                           (0, 255, 0), 1)
                        plt.imshow(capsule_region_thresholded_large_mask_removed, cmap='gray')


                    if plotfigure_tag:
                        capsule_region_thresholded_large_mask_removed = display.add_contour_to_img(capsule_region_thresholded_large_mask_removed,
                                                                                                 dyn_2d_win_l_t_removed_dark, 0,
                                                                                                 (255, 0, 0), 1)
                        plt.subplot(2, 4, 8)
                        plt.imshow(capsule_region_thresholded_large_mask_removed, cmap='gray')


                        plt.show()

    y, x = np.where(dyn_2d_win_l_t_removed_dark > 0)[0], np.where(dyn_2d_win_l_t_removed_dark > 0)[1]
    dyn_2d_win_l_t_removed_dark[y, x] = 1

    y,x = np.where(t_label_win == 1)[0], np.where(t_label_win == 1)[1]
    dyn_2d_win_l_t_removed_dark[y,x] = 1


    # --------------- Remove candidate capsule with large width --------------------------------
    if args.remove_candidate_capsule_with_large_width:


        capsule_region_thresholded_large_remove_large_width = copy.deepcopy(capsule_region_thresholded_large)

        labels, sort_list, stats, centroids, contours = array_proc.region_selection(capsule_region_thresholded_large)

        num_region = labels.max()

        for region_index in range(1, num_region + 1):
            # 构建只包含一个candidate capsule region的原图，计算该label的平均信号强度
            label_ = copy.deepcopy(labels)
            y, x = np.where(labels != region_index)
            label_[y, x] = 0
            y, x = np.where(labels == region_index)
            label_[y, x] = 1

            average_width = calculate_capsule_width(t_label_win, label_, plotfigure_tag)

            # average_signal = np.mean(dyn_2d_win[capsule_region_thresholded_large == 1])
            average_signal = np.percentile(dyn_2d_win[capsule_region_thresholded_large == 1], 75)


            # if (average_width is not None) and (average_width > args.remove_candidate_capsule_with_large_width_threshold) and radius < 20:
            #     capsule_region_thresholded_large_remove_large_width[y, x] = 0

            if (average_width is not None) and (average_width > args.remove_candidate_capsule_with_large_width_threshold):
                if radius < 20 and args.remove_bright_region and flattened_nonzero.size != 0 and (best_roi is not None) and average_signal > 1.05 * dyn_2d_win_l_t_ROI_mean:
                    capsule_region_thresholded_large_remove_large_width[y,x] = 0


        if plotfigure_tag:
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(capsule_region_thresholded_large, cmap='gray')

        y, x = np.where(capsule_region_thresholded_large_remove_large_width == 0)[0], np.where(capsule_region_thresholded_large_remove_large_width == 0)[1]
        capsule_region_thresholded_large[y,x] = 0

        if plotfigure_tag:
            plt.subplot(1,2,2)
            plt.imshow(capsule_region_thresholded_large, cmap='gray')
            plt.show()


    # --------------- Remove capsule region around liver rim -----------------------------------
    capsule_region_thresholded_large_erosion = copy.deepcopy(capsule_region_thresholded_large)
    if liver_rim_erosion:

        # Method1: liver mask erosion
        l_2d_win_erosion = array_proc.erosion(l_2d_win, args.liver_rim_erosion_size) #4
        # l_2d_win_erosion = array_proc.erosion(l_2d_win, max(int(0.2*radius),4))  # 6 for capsule along liver edge
        # print('Debugging: radius', radius*spacing, 'liver rim erosion 4: ', int(0.2*radius))

        # Strategy to remove whole capsule region along the liver edge

        if auto_segmentation and (not use_manual_parameters):
            # Method2: judge for each capsule region
            capsule_region_thresholded_large_erosion = remove_capsule_region_along_liver_rim2(capsule_region_thresholded_large, l_2d_win_erosion, dyn_2d_win, t_label_win, args)

        else:
            # y, x = np.where(l_2d_win_erosion == False)[0], np.where(l_2d_win_erosion == False)[1]
            #
            # capsule_region_thresholded_large_erosion[y, x] = 0

            # capsule_region_thresholded_large_erosion = array_proc.remove_small_length(capsule_region_thresholded_large_erosion,
            #                                                                   length=max(4, 0.2 * lesion_rim_sum))

            capsule_region_thresholded_large_erosion = remove_capsule_region_along_liver_rim1(capsule_region_thresholded_large, l_2d_win_erosion, length_threshold=0.95)



        capsule_liver_contour = display.add_contour_to_img(capsule_region_thresholded_large,
                                                                                      l_2d_win, 1,
                                                                                      (255, 0, 0), 1)
        capsule_region_thresholded_large_erosion_contour = display.add_contour_to_img(capsule_liver_contour, l_2d_win_erosion, 0,
                                                           (0, 0, 255), 1)

        ROI_lesion_contour = display.add_contour_to_img(dyn_2d_win,
                                                       t_label_win, 1,
                                                       (0, 255, 0), 1)
        ROI_liver_contour = display.add_contour_to_img(ROI_lesion_contour,l_2d_win, 0,
                                                                                      (255, 0, 0), 1)

        capsule_region_thresholded_large_erosion_contour_all = display.add_contour_to_img(ROI_liver_contour, l_2d_win_erosion, 0,
                                                           (0, 0, 255), 1)
    else:
        l_2d_win_erosion = copy.deepcopy(l_2d_win)

    if plotfigure_tag:
        plt.figure()
        plt.suptitle(phase)

        plt.subplot(3, 6, 1)
        plt.imshow(dyn_2d_win, cmap='gray')
        plt.title('ROI')

        dyn_2d_win_capsule_mask = display.add_contour_to_img(dyn_2d_win, capsule_mask, 1,
                                                                   (0, 0, 255), 1)

        plt.subplot(3, 6, 3)
        plt.imshow(dyn_2d_win_capsule_mask, cmap='gray')
        plt.title('capsule_mask')


        plt.subplot(3, 6, 2)
        plt.imshow(dyn_2d_win_lesion_around, cmap='gray')
        plt.title('lesion_around')

        if remove_tumor_dark:
            plt.subplot(3, 6, 3)
            plt.imshow(result, cmap='gray')
            plt.title('lesion_segment')


            dyn_2d_win_lesion_around_copy = copy.deepcopy(dyn_2d_win_lesion_around)
            # y, x = np.where(result == 0)[0], np.where(result == 0)[1]
            # dyn_2d_win_lesion_around_copy[y,x] = 0

            dyn_2d_win_lesion_around_copy = display.add_contour_to_img(dyn_2d_win_lesion_around_copy, result_large, 1,
                                                                                          (0, 0, 255), 1)

            plt.subplot(3, 6, 4)
            plt.imshow(dyn_2d_win_lesion_around_copy, cmap='gray')
            plt.title('lesion_segment_contour')

        # plt.subplot(3, 6, 3)
        # plt.imshow(gradient, cmap='gray')
        # plt.title('lesion_around_gradient')
        #
        # lesion_around_gradient_contour = display.add_contour_to_img(gradient, t_label_win, 1,
        #                                                    (0, 0, 255),
        #                                                    1)
        #
        # plt.subplot(3, 6, 4)
        # plt.imshow(lesion_around_gradient_contour, cmap='gray')
        # plt.title('lesion_around_gradient_contour')


        if padding_tag:
            plt.subplot(3, 6, 5)
            plt.imshow(dyn_2d_win_lesion_around_padding, cmap='gray')
            plt.title('lesion_around_padding')

            lesion_around_contour = display.add_contour_to_img(dyn_2d_win_lesion_around_padding, t_label_win, 1, (0, 0, 255),
                                                           1)
        else:
            lesion_around_contour = display.add_contour_to_img(dyn_2d_win_lesion_around, t_label_win, 1,
                                                               (0, 0, 255),
                                                               1)
        plt.subplot(3, 6, 6)
        plt.imshow(lesion_around_contour, cmap='gray')
        plt.title('lesion_around_contour')

        plt.subplot(3, 6, 7)
        plt.imshow(dyn_2d_win_enhanced, cmap='gray')
        plt.title('frangi_enhanced')

        dyn_2d_win_enhanced_tmp = display.add_contour_to_img(dyn_2d_win_enhanced, t_label_win, 1,
                                                             (0, 255, 0), 1)
        plt.subplot(3, 6, 8)
        plt.imshow(dyn_2d_win_enhanced_tmp, cmap='gray')
        plt.title('frangi_enhanced_with_contour')

        if frangi_thresholded_tag:
            if args.thresholdingFirstThenRingRegion:
                plt.subplot(3, 6, 9)
                plt.imshow(frangi_thresholded, cmap='gray')

                plt.title('frangi_enhanced_thresholded')

                # frangi_thresholded_remove_small = array_proc.remove_small_t(frangi_thresholded, np.int0(0.1 * radius))  # 15


                plt.subplot(3, 6, 10)
                plt.imshow(frangi_thresholded_large, cmap='gray')
                plt.title('frangi_enhanced_thresholded_large')
            else:
                plt.subplot(3, 6, 9)
                plt.imshow(capsule_region_withoutThresholding, cmap='gray')

                plt.title('frangi_enhanced_capsule_without_thresholding')

        plt.subplot(3, 6, 11)
        plt.imshow(capsule_region, cmap='gray')
        plt.title('frangi_enhanced_capsule')

        capsule_region_contour = display.add_contour_to_img(capsule_region, t_label_win, 1, (0, 255, 0), 1)

        plt.subplot(3, 6, 12)
        plt.imshow(capsule_region_contour, cmap='gray')
        plt.title('frangi_enhanced_capsule_contour')

        plt.subplot(3, 6, 13)
        plt.imshow(capsule_region_thresholded_large, cmap='gray')
        plt.title('fra_enhan_threded_capsule_threded_large')

        if liver_rim_erosion:
            plt.subplot(3, 6, 14)
            plt.imshow(capsule_region_thresholded_large_erosion_contour, cmap='gray')

            plt.title('fra_enhan_threded_capsule_threded_large_erosion_contour')

            plt.subplot(3, 6, 18)
            plt.imshow(capsule_region_thresholded_large_erosion_contour_all, cmap='gray')

            plt.title('fra_enhan_threded_capsule_threded_large_erosion_contour_all')

        plt.subplot(3, 6, 15)
        plt.imshow(capsule_region_thresholded_large_erosion, cmap='gray')

        plt.title('fra_enhan_threded_capsule_threded_large_erosion')

        # plt.subplot(3, 6, 16)
        # plt.imshow(liver_without_vessel_win, cmap='gray')
        # plt.title('liver_without_vessel_win')

        if vessel_removal:
            # remove large region on liver background to get vessel region
            liver_without_vessel_win_erosion_dilation = array_proc.erosion(liver_without_vessel_win, 3)
            liver_without_vessel_win_erosion_dilation = array_proc.dilate_size(liver_without_vessel_win_erosion_dilation, 3)

            labels, sort_list, stats, centroids, contours = array_proc.region_selection(liver_without_vessel_win_erosion_dilation)

            max_region = copy.deepcopy(liver_without_vessel_win_erosion_dilation)
            y, x = np.where(labels != 1)
            max_region[y, x] = 0
            y, x = np.where(labels == 1)
            max_region[y, x] = 1

            exist = (max_region != 0)
            max_sum = exist.sum()

            for region_index in range(2, labels.max() + 1):

                label = copy.deepcopy(labels)
                y, x = np.where(labels != region_index)
                label[y, x] = 0
                y, x = np.where(labels == region_index)
                label[y, x] = 1

                exist = (label != 0)
                label_sum = exist.sum()

                if label_sum > max_sum:
                    max_region = label

            y, x = np.where(max_region != 0)[0], np.where(max_region != 0)[1]
            liver_without_vessel_win[y, x] = 0
            # end

            plt.subplot(3, 6, 16)
            plt.imshow(liver_without_vessel_win, cmap='gray')
            plt.title('liver_without_vessel_win_remove_large')


            capsule_region_vessel_mask = display.add_contour_to_img(dyn_2d_win, capsule_region_thresholded_large_erosion, 1, (0, 255, 0), 1)
            capsule_region_vessel_mask = display.add_contour_to_img(capsule_region_vessel_mask, liver_without_vessel_win, 0, (255, 0, 0), 1)


            plt.subplot(3, 6, 17)
            plt.imshow(capsule_region_vessel_mask, cmap='gray')
            plt.title('capsule_region_vessel_mask')


        # plt.subplot(3, 6, 16)
        # plt.imshow(dyn_2d_win_skeleton, cmap='gray')
        # plt.title('lesion_ring_skeleton')
        #
        # plt.subplot(3, 6, 17)
        # plt.imshow(capsule_region_skeleton, cmap='gray')
        # plt.title('capsule_region_skeleton')
        #
        # lesion_gradient_skeleton = display.add_contour_to_img(gradient, capsule_region_skeleton, 1,
        #                                                    (255, 0, 0),
        #                                                    1)

        # plt.subplot(3, 6, 18)
        # plt.imshow(ret, cmap='gray')
        # plt.title('lesion_gradient_binary')
        #
        # plt.subplot(3, 6, 17)
        # plt.imshow(capsule_region_skeleton_gradient, cmap='gray')
        # plt.title('capsule_region_skeleton_gradient')

        plt.show()

    # calculate lesion rim length
    dyn_2d_win_skeleton = copy.deepcopy(dyn_2d_win)

    y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
    dyn_2d_win_skeleton[y, x] = 0

    t_label_win_erosion_1 = array_proc.erosion(t_label_win, 1)
    y, x = np.where(t_label_win_erosion_1 == 1)[0], np.where(t_label_win_erosion_1 == 1)[1]

    dyn_2d_win_skeleton[y, x] = 0

    y, x = np.where(dyn_2d_win_skeleton > 0)[0], np.where(dyn_2d_win_skeleton > 0)[1]

    dyn_2d_win_skeleton[y, x] = 1

    exist = (dyn_2d_win_skeleton != 0)
    lesion_rim_sum = exist.sum()
    # End calculate lesion rim length



    if vessel_removal:
        y, x = np.where(liver_without_vessel_win == 1)[0], np.where(liver_without_vessel_win == 1)[1]
        capsule_region_thresholded_large_erosion[y,x] = 0
        capsule_region_thresholded_large[y,x] = 0

        capsule_region_vessel_mask = display.add_contour_to_img(dyn_2d_win, capsule_region_thresholded_large_erosion, 1,
                                                                (0, 255, 0), 1)
        capsule_region_vessel_mask = display.add_contour_to_img(capsule_region_vessel_mask, liver_without_vessel_win, 0,
                                                                (255, 0, 0), 1)
        if plotfigure_tag:
            plt.subplot(3, 6, 18)
            plt.imshow(capsule_region_vessel_mask, cmap='gray')
            plt.title('capsule_region_vessel_mask')
            plt.show()

    if remove_tumor_dark:
        t_label_win_ = result_large
    else:
        t_label_win_ = t_label_win

    capsule_region_thresholded_large_erosion_skeleton = get_capsule_region_skeleton(capsule_region_thresholded_large_erosion)
    exist = capsule_region_thresholded_large_erosion_skeleton != 0

    if args.data_site == 'ZheYi':
        if exist.sum() < 0.05*lesion_rim_sum:
        # if exist.sum() < 5:

            capsule_score = capsule_double_confirm(capsule_region_thresholded_large_erosion, dyn_2d_win, t_label_win, \
                                                   t_label_win_, logging_tag, plotfigure_tag, remove_tumor_dark,args)
        else:
            capsule_score = capsule_double_confirm(capsule_region_thresholded_large, dyn_2d_win, t_label_win, t_label_win_, \
                                                   logging_tag, plotfigure_tag, remove_tumor_dark,args)

    if args.data_site == 'ZhongShan' or args.data_site == 'SuZhou' or args.data_site == 'PHC':
        if exist.sum() < 0.1 * lesion_rim_sum:

        # capsule_region_skeleton = get_capsule_region_skeleton(capsule_region_thresholded_large_erosion)
        # exist = (capsule_region_skeleton != 0)
        # capsule_skeleton_sum = exist.sum()
        #
        # capsule_score = capsule_skeleton_sum/lesion_rim_sum
            if args.remove_bright_region:
                capsule_score = capsule_double_confirm0(capsule_region_thresholded_large_erosion, dyn_2d_win, t_label_win, \
                                                        t_label_win_, l_2d_win, l_2d_win_erosion,
                                                        dyn_2d_win_l_t_removed_dark, logging_tag, plotfigure_tag, id, layer,
                                                        phase, \
                                                        statisticType=args.statisticType, args=args, best_roi_mask=None) # statisticType='percentile' or 'mean'
            else:
                capsule_score = capsule_double_confirm0(capsule_region_thresholded_large_erosion, dyn_2d_win, t_label_win, \
                                                        t_label_win_, l_2d_win, l_2d_win_erosion, dyn_2d_win_l_t_removed_dark, logging_tag, plotfigure_tag, id, layer, phase, \
                                                        statisticType=args.statisticType,args=args, best_roi_mask=None)
            # capsule_score = capsule_double_confirm(capsule_region_thresholded_large_erosion, dyn_2d_win, t_label_win, \
            #                                        t_label_win_, logging_tag, plotfigure_tag, remove_tumor_dark, args)

        else:
            if args.remove_bright_region:
                capsule_score = capsule_double_confirm0(capsule_region_thresholded_large, dyn_2d_win, t_label_win, \
                                                        t_label_win_, l_2d_win, l_2d_win_erosion,
                                                        dyn_2d_win_l_t_removed_dark, logging_tag, plotfigure_tag, id,
                                                        layer, phase, \
                                                        statisticType=args.statisticType, args=args, best_roi_mask=None)
            else:
                capsule_score = capsule_double_confirm0(capsule_region_thresholded_large, dyn_2d_win, t_label_win, \
                                                        t_label_win_, l_2d_win, l_2d_win_erosion, dyn_2d_win_l_t_removed_dark, logging_tag, plotfigure_tag, id, layer, phase, \
                                                        statisticType=args.statisticType,args=args, best_roi_mask=None)
            # capsule_score = capsule_double_confirm(capsule_region_thresholded_large, dyn_2d_win, t_label_win,
            #                                        t_label_win_, \
            #                                        logging_tag, plotfigure_tag, remove_tumor_dark, args)

    return capsule_score




def mean_intensity(img):
    exist = (img != 0)
    if exist.sum() == 0:
        mean_value = 0
    else:
        mean_value = img.sum() / exist.sum()
    return mean_value

def need_use_liver_mask(region, l_2d_win, use_liver_mask):
    region_result = copy.deepcopy(region)
    if(use_liver_mask):
        y, x = np.where(l_2d_win == 0)[0], np.where(l_2d_win == 0)[1]
        region_result[y, x] = 0

    return region_result


import copy
def Capsule_integration(dyn_2d_win, t_label_win, l_2d_win, use_liver_mask, radius,plotfigure_tag):
    dilate_size = 3

    # remove very dark area within the liver around tumor
    dyn_2d_win_normalized = copy.deepcopy(dyn_2d_win)
    cv2.normalize(dyn_2d_win, dyn_2d_win_normalized, 0, 255, cv2.NORM_MINMAX)
    # result = cv2.adaptiveThreshold(dyn_2d_win_normalized.astype(np.uint8), 255,
    #                                       cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
    #                                       2 * np.int0(0.5 * min(dyn_2d_win.shape[0], dyn_2d_win.shape[1])) + 1,
    #                                       0)  # min(35, 2*np.int0(0.5*radius)+1)
    ret, result = cv2.threshold(dyn_2d_win_normalized.astype(np.uint8), 140, 255, cv2.THRESH_BINARY) # 140, #100
    result = array_proc.erosion(result, 1)
    result = array_proc.dilate_size(result, 1)
    result = array_proc.remove_small_t(result, 0.3 * min(dyn_2d_win.shape[0], dyn_2d_win.shape[1])*min(dyn_2d_win.shape[0], dyn_2d_win.shape[1]))
    result = array_proc.hole_fill(result)
    # avoid tumor area
    y,x = np.where(t_label_win == 1)[0], np.where(t_label_win == 1)[1]
    result[y, x] = 0

    y, x = np.where(result == 0)[0], np.where(result == 0)[1]
    dyn_2d_win_remove_dark = copy.deepcopy(dyn_2d_win)
    dyn_2d_win_remove_dark[y, x] = 0

    if plotfigure_tag:
        plt.figure()

        dyn_2d_win_liver_contour = copy.deepcopy(dyn_2d_win)
        dyn_2d_win_liver_contour = display.add_contour_to_img(dyn_2d_win_liver_contour, l_2d_win, 1,
                                                              (0, 255, 0),
                                                              1)

        plt.subplot(3, 6, 1)
        plt.imshow(dyn_2d_win, cmap='gray')
        plt.title('dyn_2d_win')

        plt.subplot(3, 6, 2)
        plt.imshow(dyn_2d_win_liver_contour, cmap='gray')
        plt.title('dyn_2d_win_liver_contour')

    # remove blood vessel
    gradient = img_gradient_ROI(dyn_2d_win, t_label_win, radius)

    dyn_2d_win = gradient

    region1 = copy.deepcopy(dyn_2d_win)
    t_label_win_erosion = array_proc.erosion(t_label_win, 2)
    y, x = np.where(t_label_win_erosion == 0)[0], np.where(t_label_win_erosion == 0)[1]
    region1[y, x] = 0


    region2 = copy.deepcopy(dyn_2d_win)
    y, x = np.where(t_label_win == 0)[0], np.where(t_label_win == 0)[1]
    region2[y, x] = 0
    y, x = np.where(t_label_win_erosion == 1)[0], np.where(t_label_win_erosion == 1)[1]
    region2[y, x] = 0


    region3 = copy.deepcopy(dyn_2d_win)
    t_label_win_dilate1 = array_proc.dilate_size(t_label_win, dilate_size)
    y, x = np.where(t_label_win_dilate1 == 0)[0], np.where(t_label_win_dilate1 == 0)[1]
    region3[y, x] = 0
    y, x = np.where(t_label_win == 1)[0], np.where(t_label_win == 1)[1]
    region3[y, x] = 0

    region4 = copy.deepcopy(dyn_2d_win)
    t_label_win_dilate2 = array_proc.dilate_size(t_label_win, dilate_size*2)
    y, x = np.where(t_label_win_dilate2 == 0)[0], np.where(t_label_win_dilate2 == 0)[1]
    region4[y, x] = 0
    y, x = np.where(t_label_win_dilate1 == 1)[0], np.where(t_label_win_dilate1 == 1)[1]
    region4[y, x] = 0

    region5 = copy.deepcopy(dyn_2d_win)
    t_label_win_dilate3 = array_proc.dilate_size(t_label_win, dilate_size*3)
    y, x = np.where(t_label_win_dilate3 == 0)[0], np.where(t_label_win_dilate3 == 0)[1]
    region5[y, x] = 0
    y, x = np.where(t_label_win_dilate2 == 1)[0], np.where(t_label_win_dilate2 == 1)[1]
    region5[y, x] = 0

    region6 = copy.deepcopy(dyn_2d_win)
    t_label_win_dilate4 = array_proc.dilate_size(t_label_win, dilate_size*4)
    y, x = np.where(t_label_win_dilate4 == 0)[0], np.where(t_label_win_dilate4 == 0)[1]
    region6[y, x] = 0
    y, x = np.where(t_label_win_dilate3 == 1)[0], np.where(t_label_win_dilate3 == 1)[1]
    region6[y, x] = 0

    region1 = need_use_liver_mask(region1, l_2d_win, use_liver_mask)
    region2 = need_use_liver_mask(region2, l_2d_win, use_liver_mask)
    region3 = need_use_liver_mask(region3, l_2d_win, use_liver_mask)
    region4 = need_use_liver_mask(region4, l_2d_win, use_liver_mask)
    region5 = need_use_liver_mask(region5, l_2d_win, use_liver_mask)
    region6 = need_use_liver_mask(region6, l_2d_win, use_liver_mask)

    exist1 = (l_2d_win != 0)
    exist2 = (dyn_2d_win_remove_dark != 0)
    if exist2.sum() > exist1.sum()*1/3:
        region1 = need_use_liver_mask(region1, dyn_2d_win_remove_dark, use_liver_mask)
        region2 = need_use_liver_mask(region2, dyn_2d_win_remove_dark, use_liver_mask)
        region3 = need_use_liver_mask(region3, dyn_2d_win_remove_dark, use_liver_mask)
        region4 = need_use_liver_mask(region4, dyn_2d_win_remove_dark, use_liver_mask)
        region5 = need_use_liver_mask(region5, dyn_2d_win_remove_dark, use_liver_mask)
        region6 = need_use_liver_mask(region6, dyn_2d_win_remove_dark, use_liver_mask)

    mean1 = mean_intensity(region1)
    mean2 = mean_intensity(region2)
    mean3 = mean_intensity(region3)
    mean4 = mean_intensity(region4)
    mean5 = mean_intensity(region5)
    mean6 = mean_intensity(region6)

    mean_values = [mean1, mean2, mean3, mean4, mean5, mean6]
    # mean_max_region = np.argmax(mean_values) + 1

    mean_max_region = 1
    max_score = 0
    for i in range(1,min(round(radius/dilate_size) + 2, 5)):
    # for i in range(1, 5):
        mean_value = mean_values[i]
        mean_value_inside = mean_values[i-1]
        mean_value_outside = mean_values[i+1]
        if mean_value > mean_value_inside and mean_value > mean_value_outside:

            score = max(mean_value/mean_value_inside, mean_value/mean_value_outside)
            if score > max_score:
                mean_max_region = i + 1
                max_score = score


    t_label_win_dilate1 = need_use_liver_mask(t_label_win_dilate1, l_2d_win, use_liver_mask)
    t_label_win_dilate2 = need_use_liver_mask(t_label_win_dilate2, l_2d_win, use_liver_mask)
    t_label_win_dilate3 = need_use_liver_mask(t_label_win_dilate3, l_2d_win, use_liver_mask)
    t_label_win_dilate4 = need_use_liver_mask(t_label_win_dilate4, l_2d_win, use_liver_mask)

    if exist2.sum() > exist1.sum() * 1 / 3:
        t_label_win_dilate1 = need_use_liver_mask(t_label_win_dilate1, dyn_2d_win_remove_dark, use_liver_mask)
        t_label_win_dilate2 = need_use_liver_mask(t_label_win_dilate2, dyn_2d_win_remove_dark, use_liver_mask)
        t_label_win_dilate3 = need_use_liver_mask(t_label_win_dilate3, dyn_2d_win_remove_dark, use_liver_mask)
        t_label_win_dilate4 = need_use_liver_mask(t_label_win_dilate4, dyn_2d_win_remove_dark, use_liver_mask)


    if plotfigure_tag:

        dyn_2d_win_contour = copy.deepcopy(dyn_2d_win)
        plt.subplot(3, 6, 3)
        plt.imshow(dyn_2d_win_contour, cmap='gray')

        dyn_2d_win_contour = display.add_contour_to_img(dyn_2d_win_contour, t_label_win, 1,
                                                                    (0, 255, 0),
                                                                    1)
        plt.subplot(3, 6, 4)
        plt.imshow(dyn_2d_win_contour, cmap='gray')
        dyn_2d_win_contour = display.add_contour_to_img(dyn_2d_win_contour, t_label_win_erosion, 0,
                                                        (0, 0, 255),
                                                        1)
        plt.subplot(3, 6, 5)
        plt.imshow(dyn_2d_win_contour, cmap='gray')
        dyn_2d_win_contour = display.add_contour_to_img(dyn_2d_win_contour, t_label_win_dilate1, 0,
                                                        (255, 0, 0),
                                                        1)
        plt.subplot(3, 6, 6)
        plt.imshow(dyn_2d_win_contour, cmap='gray')

        if radius >=2*dilate_size:
            dyn_2d_win_contour = display.add_contour_to_img(dyn_2d_win_contour, t_label_win_dilate2, 0,
                                                        (0, 255, 0),
                                                        1)
            plt.subplot(3, 6, 7)
            plt.imshow(dyn_2d_win_contour, cmap='gray')
        if radius >=3*dilate_size:
            dyn_2d_win_contour = display.add_contour_to_img(dyn_2d_win_contour, t_label_win_dilate3, 0,
                                                        (0, 0, 255),
                                                        1)
            plt.subplot(3, 6, 8)
            plt.imshow(dyn_2d_win_contour, cmap='gray')
        if radius >= 4*dilate_size:
            dyn_2d_win_contour = display.add_contour_to_img(dyn_2d_win_contour, t_label_win_dilate4, 0,
                                                        (0, 0, 255),
                                                        1)
            plt.subplot(3, 6, 9)
            plt.imshow(dyn_2d_win_contour, cmap='gray')


        plt.title('dyn_2d_win_contour')

        plt.text(5, 20, str(mean_max_region),
                 fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

        plt.text(5, 40, str(round(max_score, 2)),
                 fontdict=dict(fontsize=12, color='r', family='monospace', weight='bold'))

        # plt.subplot(3, 6, 13)
        # plt.imshow(result, cmap='gray')
        # plt.title('dyn_2d_win_segmented')

        # plt.subplot(3, 6, 14)
        # plt.imshow(dyn_2d_win_remove_dark, cmap='gray')
        # plt.title('dyn_2d_win_remove_dark')


        plt.show()



    return max_score




        