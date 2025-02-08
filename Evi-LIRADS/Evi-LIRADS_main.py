import matplotlib.pyplot as plt
import numpy as np
import cv2
import math

import argparse
import yaml
import os

import time
import cProfile
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


import file_read
import array_proc
import csv_writer

from APHE_testing import APHE_Washout_ZhongShan
from Capsule import liver_vessel_removal
from Capsule import Capsule_ZhongShan2, capsule_parameters
from calculate_washout_score import Washout_Score
from judge_progressive_enhancement import if_progressive_enhancement


def main():
    # --------------------------------------------------------- Functions ------------------------------------------

    def parallel_img_array_read(name_dic):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {key: executor.submit(file_read.img_array_read, path) for key, path in name_dic.items()}
            results = {key: future.result() for key, future in futures.items()}
        return results

    def read_img_mask_ZhongShan(path, file, data_site, feature, testing_group):
        name_dic = file_read.img_path_read_ZhongShan(file, path, data_site, feature=feature, testing_group=testing_group)

        results = parallel_img_array_read(name_dic)

        pre_array = results['pre'][0]
        A_array = results['A'][0]
        V_array = results['V'][0]
        D_array = results['D'][0]

        t_pre_array = results['t_pre'][0]
        t_A_array = results['t_A'][0]
        t_V_array = results['t_V'][0]
        t_D_array, spacing = results['t_D']

        l_pre_array = results['l_pre'][0]
        l_A_array = results['l_A'][0]
        l_V_array = results['l_V'][0]
        l_D_array = results['l_D'][0]

        return pre_array, A_array, V_array, D_array, t_pre_array,t_A_array, t_V_array, t_D_array, l_pre_array, l_A_array, l_V_array, l_D_array, spacing


    def get_tumor_layers_ZhongShan(img_array,t_array, l_array):
        t_array_pre = array_proc.remove_t_1(t_array)  # remove label which is > 1

        img_layers, t_layers, l_layers, tz_layers = array_proc.get_layers_single_phase(img_array,t_array_pre, l_array)  # remove layers which doesn't have label = 1

        return img_layers, t_layers, l_layers, tz_layers


    def single_layer_ZhongShan(pre_layers, A_layers, V_layers, D_layers, t_pre_layers, t_A_layers, t_V_layers, t_D_layers,  l_pre_layers, l_A_layers, l_V_layers, l_D_layers, layer_index):
        pre_array_2d = pre_layers[layer_index[0], :, :]
        A_array_2d = A_layers[layer_index[1], :, :]
        V_array_2d = V_layers[layer_index[2], :, :]
        D_array_2d = D_layers[layer_index[3], :, :]

        t_pre_array_2d = t_pre_layers[layer_index[0], :, :]
        t_A_array_2d = t_A_layers[layer_index[1], :, :]
        t_V_array_2d = t_V_layers[layer_index[2], :, :]
        t_D_array_2d = t_D_layers[layer_index[3], :, :]

        l_pre_array_2d = l_pre_layers[layer_index[0], :, :]
        l_A_array_2d = l_A_layers[layer_index[1], :, :]
        l_V_array_2d = l_V_layers[layer_index[2], :, :]
        l_D_array_2d = l_D_layers[layer_index[3], :, :]


        return pre_array_2d, A_array_2d, V_array_2d, D_array_2d, t_pre_array_2d, t_A_array_2d, t_V_array_2d,t_D_array_2d, l_pre_array_2d, l_A_array_2d, l_V_array_2d,l_D_array_2d

    def tumor_layer_pre_processing(t_array_2d, smallest_region_pixels):
        t_2d_pre1 = array_proc.remove_small_t(t_array_2d, smallest_region_pixels)
        t_2d_pre2 = array_proc.hole_fill(t_2d_pre1)
        t_array_2d_pre = array_proc.label(t_2d_pre2)  # get lesion connected region

        return t_array_2d_pre

    def tumor_details(t_array_2d_pre,spacing, tz_layers, tumor_layer_index, Lesion_details, Lesion_layers):
        Lesion_layers.append(tz_layers[tumor_layer_index])

        t_array_2d_lesions = np.uint8(t_array_2d_pre)

        ret, binary = cv2.threshold(t_array_2d_lesions, 0, 1, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary,
                                                        cv2.RETR_LIST,
                                                        cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            center, radius = cv2.minEnclosingCircle(contours[0])
            # center = np.int0(center)
            diameter_cm = round((radius * spacing[0] * 0.1*2), 1)

            # print('i is: ', i)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)


            for l in range(0, num_labels - 1):

                lesion_details = [tz_layers[tumor_layer_index] + 1, num_labels - 1, l + 1, round(centroids[l + 1][0], 2),
                                  round(centroids[l + 1][1], 2), round(stats[l + 1][4], 2), round(radius * 2, 2), diameter_cm]
                # print(lesion_details)
                Lesion_details.append(lesion_details)
                #fileheader = ['layer_ID', 'num_lesion', 'lesion_ID', 'lesion_pos_x', 'lesion_pos_y', 'lesion_area', 'diameter_pixels', 'diameter_cm']
        else:
            lesion_details = [tz_layers[tumor_layer_index] + 1, 0, 0, 0,
                              0, 0, 0, 0]
            Lesion_details.append(lesion_details)
        return Lesion_details, Lesion_layers



    def combine_liver_tumor(l_2d_win, t_label_win):
        import copy
        l_2d_win_combined = copy.deepcopy(l_2d_win)
        y, x = np.where(t_label_win == 1)[0], np.where(t_label_win == 1)[1]
        l_2d_win_combined[y, x] = 1

        return l_2d_win_combined



    def get_DCE_data_ZhongShan(path, file, data_site, feature, testing_group):
        # Read image and mask
        pre_array, A_array, V_array, D_array, t_pre_array, t_A_array, t_V_array, t_D_array, \
        l_pre_array, l_A_array, l_V_array, l_D_array, spacing = read_img_mask_ZhongShan(
            path, file, data_site, feature, testing_group)


        arrays = [(pre_array, t_pre_array, l_pre_array),
                  (A_array, t_A_array, l_A_array),
                  (V_array, t_V_array, l_V_array),
                  (D_array, t_D_array, l_D_array)]

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda arr: get_tumor_layers_ZhongShan(arr[0], arr[1], arr[2]), arrays))

        # 将结果解包
        (pre_layers, t_pre_layers, l_pre_layers, tz_layers_pre), (A_layers, t_A_layers, l_A_layers, tz_layers_A), (V_layers, t_V_layers, l_V_layers, tz_layers_V), (D_layers, t_D_layers, l_D_layers, tz_layers_D) = results

        return pre_layers, A_layers, V_layers, D_layers, t_pre_layers, t_A_layers, t_V_layers, t_D_layers, l_pre_layers, l_A_layers, l_V_layers, l_D_layers, tz_layers_pre, tz_layers_A, tz_layers_V, tz_layers_D, spacing


    def pre_processing_single_layer_ZhongShan(pre_layers, A_layers, V_layers, D_layers, t_pre_layers, t_A_layers, t_V_layers,
                                   t_D_layers, l_pre_layers, l_A_layers, l_V_layers, l_D_layers, layer_index, file):
        pre_array_2d, A_array_2d, V_array_2d, D_array_2d, t_pre_array_2d, t_A_array_2d, t_V_array_2d, t_D_array_2d, \
            l_pre_array_2d, l_A_array_2d, l_V_array_2d, l_D_array_2d = \
            single_layer_ZhongShan(pre_layers, A_layers, V_layers, D_layers, t_pre_layers, t_A_layers, t_V_layers,
                                   t_D_layers,  l_pre_layers, l_A_layers, l_V_layers, l_D_layers, layer_index)


        t_pre_array_2d_pre = tumor_layer_pre_processing(t_pre_array_2d, args.smallest_region_pixels_ZhongShan)
        t_A_array_2d_pre = tumor_layer_pre_processing(t_A_array_2d, args.smallest_region_pixels_ZhongShan)
        t_V_array_2d_pre = tumor_layer_pre_processing(t_V_array_2d, args.smallest_region_pixels_ZhongShan)
        t_D_array_2d_pre = tumor_layer_pre_processing(t_D_array_2d, args.smallest_region_pixels_ZhongShan)

        t_array_2d_max = min(t_pre_array_2d_pre.max(), t_A_array_2d_pre.max(), t_V_array_2d_pre.max(),
                             t_D_array_2d_pre.max())

        # liver mask preprocessing
        l_pre_array_2d = array_proc.hole_fill_liver_mask(l_pre_array_2d)
        l_pre_array_2d = combine_liver_tumor(l_pre_array_2d, t_pre_array_2d_pre)

        l_A_array_2d = array_proc.hole_fill_liver_mask(l_A_array_2d)
        l_A_array_2d = combine_liver_tumor(l_A_array_2d, t_A_array_2d_pre)


        l_V_array_2d = array_proc.hole_fill_liver_mask(l_V_array_2d)
        l_V_array_2d = combine_liver_tumor(l_V_array_2d, t_V_array_2d_pre)

        l_D_array_2d = array_proc.hole_fill_liver_mask(l_D_array_2d)
        l_D_array_2d = combine_liver_tumor(l_D_array_2d, t_D_array_2d_pre)

        return pre_array_2d, A_array_2d, V_array_2d, D_array_2d, t_pre_array_2d_pre, t_A_array_2d_pre, t_V_array_2d_pre, t_D_array_2d_pre, l_pre_array_2d, l_A_array_2d, l_V_array_2d, l_D_array_2d, t_array_2d_max

    def single_layer_single_phase_ZhongShan(pre_layers, t_pre_layers, l_pre_layers,layer_index):
        pre_array_2d = pre_layers[layer_index, :, :]
        t_pre_array_2d = t_pre_layers[layer_index, :, :]
        l_pre_array_2d = l_pre_layers[layer_index, :, :]

        return pre_array_2d,t_pre_array_2d, l_pre_array_2d

    def pre_processing_single_layer_single_phase_ZhongShan(pre_layers, t_pre_layers,l_pre_layers, layer_index, file):
        pre_array_2d, t_pre_array_2d, l_pre_array_2d = single_layer_single_phase_ZhongShan(pre_layers, t_pre_layers, l_pre_layers, layer_index)


        t_pre_array_2d_pre = tumor_layer_pre_processing(t_pre_array_2d, args.smallest_region_pixels_ZhongShan)
        # Steps:
        # morphology.remove_small_objects
        # fill holes
        # get connected regions(In fact, just one connected region)

        # liver mask preprocessing
        l_pre_array_2d = array_proc.hole_fill_liver_mask(l_pre_array_2d)
        l_pre_array_2d = combine_liver_tumor(l_pre_array_2d, t_pre_array_2d_pre)

        return pre_array_2d, t_pre_array_2d_pre, l_pre_array_2d



    def _parse_args():
        # Do we have a config file to parse?
        args_config, remaining = config_parser.parse_known_args()
        if args_config.config:
            with open(args_config.config, 'r') as f:
                cfg = yaml.safe_load(f)
                parser.set_defaults(**cfg)

        # The main arg parser parses the rest of the args, the usual
        # defaults will have been overridden if config file specified.
        args = parser.parse_args(remaining)

        # Cache the args as a text string to save them in the output dir later
        args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
        return args, args_text


    for testing_group in range(1,2):
        for data_site in ['ZS', 'SZ']:

            print('################################################ testing group: ', testing_group, 'site:', data_site)
            start_time_all = time.time()

            isServer = False

            # APHE_Score_Threshold = 1.114
            # Washout_Score_Threshold = 1.0267 # 1.026717225	95%CI: 1.026641247	1.026793202
            Capsule_Score_Threshold = 0.41
            Not_All_Slices = False

            Calculate_APHE = True
            Calculate_Washout = True
            Calculate_Capsule = True  #记得更改 get_DCE_data_ZhongShan里面的feature为Capsule (PHC)


            auto_segmentation = True

            if auto_segmentation:
                registered = True
            else:
                registered = False

            plus_manual = True


            WashInFirst = False
            Capsule_Updated = True

            # 1） Wash In
            # scoreAThreshold=0.95;

            # 2）V：compare enhancedTumor with Liver
            V_TumorBright_Liver_intensity_ratio_threshold = 1.50

            # 3）Compare signal intensity of TumorV to TumorA using the enhanced tumor region
            liver_enhancement_k = 1.005
            # if V_2d_win_bright_mean > A_2d_win_bright_mean and V_2d_win_bright_mean / V_around_mean > A_2d_win_bright_mean / (
            #         A_around_mean * liver_enhancement_k) and (not whole_enhancement):
            V_bright_area_ratio_A_threshold = 0.05


            # 4）signal intensity ratio of tumor to liver, compare V to A

            VOI = False
            # file_read.py中需要相应修改

            resampled = False
            rescaled = False

            # external = True # Capsule.py also needs to be changed

            # show_middle_result = False
            plotfigure_tag = False
            detailplot_tag = False

            plotTimeIntensityCurve = False

            ############################
            write_result = True
            logging_tag = True

            ################################
            vessel_removal = False
            ###############################

            justTesting = False
            justTrainingValidation = False


            # The first arg parser parses out only the --config argument, this argument is used to
            # load a yaml file containing key-values that override the defaults for the main parser below
            config_parser = parser = argparse.ArgumentParser(description='LIRADS feature recognition', add_help=False)
            parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                                help='YAML config file specifying default arguments')
            parser = argparse.ArgumentParser(description='LIRADS feature recognition')

            parser.add_argument('--data_site', default=data_site, type=str)


            parser.add_argument('--path', default='' + '/', type=str) # folder path of the images



            parser.add_argument('--layer_match_method', action='store_true', default=2,
                                help='# 1: all from first tumor slice # 2: use same slices first')
            parser.add_argument('--consider_pre', action='store_true', default=True,
                                help='')
            parser.add_argument('--linear_fitting', action='store_true', default=False,
                                help='')
            parser.add_argument('--linear_fitting_lamda', action='store_true', default=0.1,
                                help='')

            # 0. ROI
            parser.add_argument('--element_size', action='store_true', default=25,
                                help='# Get region around tumor')


            # parameters for pre-processing

            parser.add_argument('--smallest_region_pixels_ZhongShan', action='store_true', default=3,
                                    help='')


            # 1. parameters for APHE
            parser.add_argument('--around_dilate_size', action='store_true', default=10,
                                help='')

            # 2. parameters for Washout
            parser.add_argument('--rescaled', action='store_true', default=rescaled,
                                    help='default: False')
            if auto_segmentation:
                parser.add_argument('--WashoutCompareTumorAtoTumorPre', action='store_true', default=True,
                                    help='default: False')
            else:
                parser.add_argument('--WashoutCompareTumorAtoTumorPre', action='store_true', default=True,
                                    help='default: False')

            parser.add_argument('--compareTumorVwithTumorA_normalizedByLiver', action='store_true', default=True,
                                    help='default: True')



            parser.add_argument('--scoreAThreshold', action='store_true', default=0.95, help='default: 0.95') # scoreAThreshold：用于是否WashIn的判断

            # 3. parameters for Capsule
            parser.add_argument('--padding_tag', action='store_true', default=True,
                                help='')


            parser.add_argument('--liver_rim_erosion', action='store_true', default=False,
                                        help='default:False')

            parser.add_argument('--frangi_thresholded_tag', action='store_true', default=True,
                                help='# ZhongShan: True; SuZhou: False')

            parser.add_argument('--remove_tumor_dark', action='store_true', default=False,
                                help='')
            parser.add_argument('--frangi_threshold', action='store_true', default=65,
                                help='')



            # parameters for distance transform
            parser.add_argument('--dis_threshold', action='store_true', default=0.85,
                                help='# used in function - tumor around liver, not used in the main function right now')


            parser.add_argument('--inside_threshold', action='store_true', default=0.9,
                                help='')
            parser.add_argument('--dilate_k', action='store_true', default=0.2,
                                help='')

            parser.add_argument('--outside_threshold', action='store_true', default=0,
                                help='# tumor around liver')


            ########################################################################################
            args, args_text = _parse_args()

            #######################################################################################
            dirs = sorted(os.listdir(args.path))

            if not os.path.exists('./result/Time'):
                os.mkdir('./result/Time')
            result_dir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())

            if Calculate_APHE:
                if not os.path.exists('./result/APHE'):
                    os.mkdir('./result/APHE')

                result_dir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())


            if Calculate_Washout:
                if not os.path.exists('./result/Washout'):
                    os.mkdir('./result/Washout')

                result_dir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())


            if Calculate_Capsule:
                if not os.path.exists('./result/Capsule'):
                    os.mkdir('./result/Capsule')

                result_dir = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())



            if write_result:
                csvfilename_time = './result/Time/' + result_dir + '/' + data_site + '_Time.csv'
                os.makedirs(os.path.dirname(csvfilename_time), exist_ok=True)
                csvfile_all_Time = open(csvfilename_time, 'w', newline='' '')
                fileheader_all = ['sample_ID', 'running time']
                csv_writer.write_to_csv(csvfile_all_Time, fileheader_all)

                if Calculate_APHE:
                    csvfilename = './result/APHE/' + result_dir + '/' + data_site + '_APHE'

                    os.makedirs(os.path.dirname(csvfilename), exist_ok=True)
                    csvfile_all_APHE = open(csvfilename, 'w', newline='' '')

                    fileheader_all = ['sample_ID', 'final_score', 'at layer', 'max diameter']
                    csv_writer.write_to_csv(csvfile_all_APHE, fileheader_all)

                if Calculate_Washout:
                    csvfilename = './result/Washout/' + result_dir + '/' + data_site + '_Washout'

                    os.makedirs(os.path.dirname(csvfilename), exist_ok=True)
                    csvfile_all_Washout = open(csvfilename, 'w', newline='' '')
                    fileheader_all = ['sample_ID', 'final_score', 'at layer', 'max diameter']
                    csv_writer.write_to_csv(csvfile_all_Washout, fileheader_all)

                if Calculate_Capsule:
                    csvfilename = './result/Capsule/' + result_dir + '/' + data_site + '_Capsule'

                    os.makedirs(os.path.dirname(csvfilename), exist_ok=True)
                    csvfile_all_Capsule = open(csvfilename, 'w', newline='' '')
                    fileheader_all = ['sample_ID', 'final_score', 'at layer', 'max diameter']
                    csv_writer.write_to_csv(csvfile_all_Capsule, fileheader_all)

            end_time_all = time.time()
            time_all = end_time_all - start_time_all
            csv_writer.write_to_csv(csvfile_all_Time, ['Preparation Time', time_all])

            for file_index in range(0, len(dirs)):

                file = dirs[file_index]
                print(len(dirs), "file ", file_index, ': ', file)

                start_time_each = time.time()


                if write_result:
                    if Calculate_APHE:
                        csvfilename = './result/APHE/' + result_dir + '/' + data_site + '_APHE_' + file
                        csvfile_APHE = open(csvfilename, 'w', newline='' '')
                        fileheader = ['phase', 'layer_ID', 'num_lesion', 'lesion_ID', 'lesion_pos_x', 'lesion_pos_y', 'lesion_area', 'diameter_pixels', 'diameter_cm', 'score']
                        csv_writer.write_to_csv(csvfile_APHE, fileheader)

                    if Calculate_Washout:
                        csvfilename = './result/Washout/' + result_dir + '/' + data_site + '_Washout_' + file
                        csvfile_Washout = open(csvfilename, 'w', newline='' '')
                        fileheader = ['phase', 'layer_ID', 'num_lesion', 'lesion_ID', 'lesion_pos_x', 'lesion_pos_y', 'lesion_area', 'diameter_pixels', 'diameter_cm', 'score']
                        csv_writer.write_to_csv(csvfile_Washout, fileheader)

                    if Calculate_Capsule:
                        csvfilename = './result/Capsule/' + result_dir + '/' + data_site + '_Capsule_' + file
                        csvfile_Capsule = open(csvfilename, 'w', newline='' '')
                        fileheader = ['phase', 'layer_ID', 'num_lesion', 'lesion_ID', 'lesion_pos_x', 'lesion_pos_y', 'lesion_area', 'diameter_pixels', 'diameter_cm', 'score']
                        csv_writer.write_to_csv(csvfile_Capsule, fileheader)


                pre_layers, A_layers, V_layers, D_layers, t_pre_layers, t_A_layers, t_V_layers, t_D_layers, \
                l_pre_layers, l_A_layers, l_V_layers, l_D_layers, \
                tz_layers_pre, tz_layers_A, tz_layers_V, tz_layers_D, spacing = get_DCE_data_ZhongShan(args.path, file, data_site,feature='Capsule', testing_group=testing_group)

                print('tz_layers_pre_A_V: ', tz_layers_pre, tz_layers_A, tz_layers_V)
                # Steps:
                # read .nii, images and masks
                # remove label pixels which is > 1: just consider HCC or suspected HCC
                # remove layers which doesn't have label = 1: select tumor slices
                # Output:
                # tz_layers_X: true slice index, like [46 47 48]



                if args.layer_match_method == 1:
                    num_layers = min(pre_layers.shape[0], A_layers.shape[0], V_layers.shape[0], D_layers.shape[0])

                if args.layer_match_method == 2:

                    layers_pre = tz_layers_pre
                    layers_A = tz_layers_A

                    layers_V = tz_layers_V

                    num_layers = len(layers_pre)

                print('Total layers: ', num_layers)
                # %%
                if num_layers > 0:
                    if write_result:
                        scores_APHE = []

                        scores1 = []
                        diameters = []

                        Lesion_details = []
                        Lesion_layers = []


                    tumorPres = []
                    tumorAs = []
                    liverAs = []
                    ratioTumorAtoPre = []
                    ratioScoreAtoPre = []
                    scoreAs = []

                    TumorVBrightMeans = []
                    TumorABrightMeans = []
                    TumorVDarkMeans = []
                    TumorADarkMeans = []

                    pre_t_label_winS = []

                    Whole_enhancement = []
                    Enhancement_area_ratio_A = []
                    Enhancement_area_ratio_V = []
                    V_bright_area_ratio_A_all = []
                    V_bright_around_all = []

                    if Calculate_APHE or Calculate_Washout:
                        for i in range(0, num_layers):

                            if args.layer_match_method == 2:
                                layer_pre = layers_pre[i]
                                layer_A = layers_A[i]

                                layer_V = layers_V[i]
                                # layer_X: true slice index, like 46
                            if logging_tag:
                                if args.layer_match_method == 1:
                                    print('layer num: ', i + 1)
                                if args.layer_match_method == 2:
                                    print('layer num: ', layer_pre + 1, layer_A + 1)



                            if args.layer_match_method == 1:
                                layer_index = [i, i, i, i]

                                pre_array_2d, A_array_2d, V_array_2d, D_array_2d, t_pre_array_2d_pre, t_A_array_2d_pre, t_V_array_2d_pre, t_D_array_2d_pre, \
                                l_pre_array_2d, l_A_array_2d, l_V_array_2d, l_D_array_2d, t_array_2d_max = \
                                    pre_processing_single_layer_ZhongShan(pre_layers, A_layers, V_layers, D_layers,
                                                                          t_pre_layers, t_A_layers, t_V_layers, t_D_layers,
                                                                          l_pre_layers, l_A_layers, l_V_layers, l_D_layers,
                                                                          layer_index, file)

                            if args.layer_match_method == 2:
                                if not registered:
                                    layer_index = [tz_layers_pre.tolist().index(layer_pre),
                                                   tz_layers_A.tolist().index(layer_A)]
                                else:
                                    layer_index = [tz_layers_pre.tolist().index(layer_pre), tz_layers_A.tolist().index(layer_A), tz_layers_V.tolist().index(layer_V)]


                                # layer_index, the position of layer_pre in tz_layers_pre, and position of layer_A in tz_layers_A;
                                # 46: 0; 46: 0; [0,0]
                                pre_array_2d, t_pre_array_2d_pre, l_pre_array_2d = pre_processing_single_layer_single_phase_ZhongShan(\
                                    pre_layers, t_pre_layers,l_pre_layers, layer_index[0], file)
                                A_array_2d, t_A_array_2d_pre, l_A_array_2d = pre_processing_single_layer_single_phase_ZhongShan(
                                    A_layers, t_A_layers, l_A_layers, layer_index[1], file)


                                V_array_2d, t_V_array_2d_pre, l_V_array_2d = pre_processing_single_layer_single_phase_ZhongShan(
                                    V_layers, t_V_layers, l_V_layers, layer_index[2], file)


                                t_array_2d_max = 1


                            if ((t_pre_array_2d_pre.max() == t_pre_array_2d_pre.min()) or (
                                    t_A_array_2d_pre.max() == t_A_array_2d_pre.min())):

                                if write_result:
                                    scores_APHE.append(0)
                                    diameters.append(0)
                                    ################################# Calculate APHE Score ##############################################

                                    Lesion_details, Lesion_layers = tumor_details(t_A_array_2d_pre, spacing, tz_layers_A, i,
                                                                                  Lesion_details, Lesion_layers)
                                print('No lesion larger than 1 in this layer !!')

                            else:

                                # %%
                                for j in range(1, t_array_2d_max + 1):
                                    if args.layer_match_method == 1:
                                        pre_2d_win, A_2d_win, V_2d_win, D_2d_win, pre_t_label_win, A_t_label_win, V_t_label_win, D_t_label_win, \
                                            pre_l_array_win, A_l_array_win, V_l_array_win, D_l_array_win = \
                                            array_proc.crop_ZhongShan(t_pre_array_2d_pre, t_A_array_2d_pre, t_V_array_2d_pre,
                                                                      t_D_array_2d_pre, j, pre_array_2d, A_array_2d, V_array_2d,
                                                                      D_array_2d, l_pre_array_2d,l_A_array_2d, l_V_array_2d,l_D_array_2d,
                                                                      args.element_size)  # get ROI

                                    if args.layer_match_method == 2:

                                        pre_2d_win, pre_t_label_win, pre_l_array_win = array_proc.crop_single_phase_ZhongShan\
                                            (t_pre_array_2d_pre,j,pre_array_2d, l_pre_array_2d, args.element_size)
                                        A_2d_win, A_t_label_win, A_l_array_win = array_proc.crop_single_phase_ZhongShan \
                                            (t_A_array_2d_pre, j, A_array_2d, l_A_array_2d, args.element_size)

                                        V_2d_win, V_t_label_win, V_l_array_win = array_proc.crop_single_phase_ZhongShan \
                                            (t_V_array_2d_pre, j, V_array_2d, l_V_array_2d, args.element_size)


                                        pre_t_label_winS.append(pre_t_label_win)



                                    if write_result:
                                        ################################# Calculate APHE Washout and Capsule Score ##############################################

                                        Lesion_details, Lesion_layers = tumor_details(t_A_array_2d_pre, spacing, tz_layers_A, i, Lesion_details, Lesion_layers)
                                        radius = Lesion_details[len(Lesion_details)-1][6]*0.5

                                        #---------------------------------------------------------------------------------
                                        if args.layer_match_method == 1:
                                            liver_without_vessel_V, liver_without_vessel_V_remove_large = liver_vessel_removal(V_array_2d, t_V_array_2d_pre, l_V_array_2d, radius, plotfigure_tag, vessel_removal=vessel_removal)
                                            liver_without_vessel_V_win = array_proc.crop_single_phase_single_img(t_V_array_2d_pre, j, liver_without_vessel_V, args.element_size)
                                            liver_without_vessel_V_win_remove_large = array_proc.crop_single_phase_single_img(t_V_array_2d_pre, j, liver_without_vessel_V_remove_large, args.element_size)

                                            liver_without_vessel_D, liver_without_vessel_D_remove_large = liver_vessel_removal(
                                                D_array_2d, t_D_array_2d_pre, l_D_array_2d, radius, plotfigure_tag, vessel_removal=vessel_removal)
                                            liver_without_vessel_D_win = array_proc.crop_single_phase_single_img(t_D_array_2d_pre, j,
                                                                                                                 liver_without_vessel_D,
                                                                                                                 args.element_size)
                                            liver_without_vessel_D_win_remove_large = array_proc.crop_single_phase_single_img(
                                                t_D_array_2d_pre, j, liver_without_vessel_D_remove_large, args.element_size)

                                        #-----------------------------------------------------------------------------------

                                        # 1. Calculate APHE score

                                        tumor_OTSU_bright_Pre, around_pix_img_mean_value_Pre, t_point_img_mean_value_Pre, tumor_OTSU_dark_Pre, enhancement_area_ratio_Pre, whole_enhancement_Pre = \
                                            APHE_Washout_ZhongShan(pre_2d_win, pre_t_label_win, pre_l_array_win, j,
                                                                   args.around_dilate_size, radius, 'pre',
                                                                   data_site, rescaled=args.rescaled,
                                                                   feature='APHE', A_whole_enhancement=False)

                                        tumor_OTSU_bright_A, around_pix_img_mean_value_A, t_point_img_mean_value_A, tumor_OTSU_dark_A, enhancement_area_ratio_A, whole_enhancement_A = \
                                            APHE_Washout_ZhongShan(A_2d_win, A_t_label_win, A_l_array_win, j,
                                                                   args.around_dilate_size, radius, 'A', data_site,
                                                                   rescaled=args.rescaled, feature='APHE',
                                                                   A_whole_enhancement=False)

                                        # scorePre = tumorPre / liverPre
                                        # scoreA = tumorA/liverA

                                        scorePre = tumor_OTSU_bright_Pre / around_pix_img_mean_value_Pre
                                        # scorePre = t_point_img_mean_value_Pre / around_pix_img_mean_value_Pre
                                        scoreA = tumor_OTSU_bright_A / around_pix_img_mean_value_A

                                        # scorePre = t_point_img_mean_value_Pre / around_pix_img_mean_value_Pre # For Comparison Experiment
                                        # scoreA = t_point_img_mean_value_A / around_pix_img_mean_value_A


                                        scorePre_Washout = t_point_img_mean_value_Pre / around_pix_img_mean_value_Pre
                                        scoreA_Washout = tumor_OTSU_bright_A / around_pix_img_mean_value_A



                                        if scorePre > 1:
                                            if args.consider_pre:
                                                if args.linear_fitting:
                                                    aphe_score = scoreA*(1-args.linear_fitting_lamda) + scoreA / scorePre * args.linear_fitting_lamda
                                                else:
                                                    aphe_score = scoreA/scorePre

                                            else:
                                                aphe_score = scoreA
                                            print('#############################  ', file, ' scorePre > 1 #####################################' )
                                        else:
                                            aphe_score = scoreA



                                        print('For APHE:')
                                        print(scoreA, aphe_score)

                                        # Calculate Washout Score
                                        #########################################################################################################
                                        # -----------------------  Compare signal intensity of TumorV to TumorA using the enhanced tumor region ---------------------------
                                        #########################################################################################################

                                        # For Washout
                                        print('For Washout: ')

                                        tumor_OTSU_bright_Pre, around_pix_img_mean_value_Pre, t_point_img_mean_value_Pre, tumor_OTSU_dark_Pre, enhancement_area_ratio_Pre, whole_enhancement_Pre = \
                                            APHE_Washout_ZhongShan(pre_2d_win, pre_t_label_win, pre_l_array_win, j,
                                                                   args.around_dilate_size, radius, 'pre',
                                                                   data_site, rescaled=args.rescaled,
                                                                   feature='Washout', A_whole_enhancement=False)

                                        tumor_OTSU_bright_A, around_pix_img_mean_value_A, t_point_img_mean_value_A, tumor_OTSU_dark_A, enhancement_area_ratio_A, whole_enhancement_A = \
                                            APHE_Washout_ZhongShan(A_2d_win, A_t_label_win, A_l_array_win, j,
                                                                   args.around_dilate_size, radius, 'A', data_site,
                                                                   rescaled=args.rescaled, feature='Washout',
                                                                   A_whole_enhancement=False)

                                        tumorPre, liverPre = t_point_img_mean_value_Pre, around_pix_img_mean_value_Pre
                                        if auto_segmentation:
                                            tumorA, liverA = tumor_OTSU_bright_A, around_pix_img_mean_value_A
                                        else:
                                            tumorA, liverA = t_point_img_mean_value_A, around_pix_img_mean_value_A


                                        V_bright_area_ratio_A, V_2d_win_bright_mean, A_2d_win_bright_mean, V_2d_win_dark_mean, A_2d_win_dark_mean, V_bright_around = \
                                            if_progressive_enhancement(A_2d_win, A_t_label_win, A_l_array_win, V_2d_win, V_t_label_win, V_l_array_win, radius, liver_enhancement_k, whole_enhancement_A, auto_segmentation, plotfigure_tag)


                                        if auto_segmentation:
                                            TumorVBrightMeans += [V_2d_win_bright_mean]
                                            TumorABrightMeans += [A_2d_win_bright_mean]
                                            TumorVDarkMeans += [V_2d_win_dark_mean]
                                            TumorADarkMeans += [A_2d_win_dark_mean]

                                            Whole_enhancement += [whole_enhancement_A]
                                            Enhancement_area_ratio_A += [enhancement_area_ratio_A]

                                            V_bright_area_ratio_A_all += [V_bright_area_ratio_A]
                                            V_bright_around_all += [V_bright_around]


                                        ##############  End V #####################



                                        scores_APHE.append(aphe_score)

                                        tumorPres += [tumorPre]
                                        tumorAs += [tumorA]
                                        liverAs += [liverA]
                                        ratioTumorAtoPre += [tumorA/tumorPre]
                                        ratioScoreAtoPre += [(tumorA / liverA) / (tumorPre / liverPre)]
                                        scoreAs += [scoreA]



                                        diameters.append(Lesion_details[len(Lesion_details)-1][7])

                                    plt.close('all')


                        print('Write feature APHE.')
                        if write_result == True and len(scores_APHE) > 0:


                            scores_APHE = [x for x in scores_APHE if math.isnan(x) == False]
                            grade = max(scores_APHE)
                            # grade1 = max(scores1)

                            max_layer = np.argmax(scores_APHE)
                            # max_layer1 = np.argmax(scores1)
                            # grade = scores
                            max_diameter = max(diameters)
                            print('############################################# APHE Final score is: ', grade,  ' ##############################')

                            if Calculate_APHE:
                                csv_writer.write_to_csv(csvfile_APHE,
                                                        ['final_score:', grade])

                                csvfile_APHE.close()
                            print('Done!')


                            if Calculate_APHE:
                                lesion_score = [file, grade, max_diameter]

                                csv_writer.write_to_csv(csvfile_all_APHE, lesion_score)
                else:
                    print('#################################################### Need to Check ##########################################')




                # Calculate based on phase V
                print('##################################################### Calculate based on phase V ... ##############################')
                num_layers = tz_layers_V.shape[0]
                # num_layers = len(layers)
                print('Total layers V: ', num_layers)

                # if feature == 'Washout':
                if Calculate_Washout:
                    print('#####################################################  For Washout: ###############################################')
                    num_layers_pre_A = len(pre_t_label_winS)
                    print('Total layers pre_A: ', num_layers_pre_A)

                    num_layers = min(num_layers, num_layers_pre_A)
                    print('Washout layers: ', num_layers)
                # %%
                if num_layers > 0:
                    if write_result:
                        scores_Washout = []
                        scores_Capsule = []
                        scoreVs = []
                        tumorVs = []
                        liverVs = []
                        diameters = []

                        Lesion_details = []
                        Lesion_layers = []


                    if Calculate_Washout:
                        for i in range(0, num_layers):
                        # for i in range(8, 9):

                            if logging_tag:
                                print('layer num: ', i + 1)

                            layer_index = i


                            V_array_2d, t_V_array_2d_pre, l_V_array_2d = pre_processing_single_layer_single_phase_ZhongShan(
                                V_layers, t_V_layers, l_V_layers, layer_index, file)
                            t_array_2d_max = 1

                            if t_V_array_2d_pre.max() == t_V_array_2d_pre.min():

                                if write_result:
                                    scores_Washout.append(0)
                                    diameters.append(0)
                                    ################################# Calculate APHE Score ##############################################

                                    if i < len(tz_layers_A):
                                        Lesion_details, Lesion_layers = tumor_details(t_A_array_2d_pre, spacing, tz_layers_A, i,
                                                                                      Lesion_details, Lesion_layers)
                                print('No lesion larger than 1 in this layer !!')
                            else:

                                # %%
                                for j in range(1, t_array_2d_max + 1):
                                    # %%

                                    V_2d_win, V_t_label_win, V_l_array_win = array_proc.crop_single_phase_ZhongShan \
                                        (t_V_array_2d_pre, j, V_array_2d, l_V_array_2d, args.element_size)

                                    if write_result:
                                        ################################# Calculate APHE Washout and Capsule Score ##############################################

                                        Lesion_details, Lesion_layers = tumor_details(t_V_array_2d_pre, spacing, tz_layers_V, i,
                                                                                      Lesion_details, Lesion_layers)

                                        radius = Lesion_details[len(Lesion_details) - 1][6] * 0.5

                                        #---------------------------------------------------------------------------------
                                        liver_without_vessel_V, liver_without_vessel_V_remove_large = liver_vessel_removal(V_array_2d, t_V_array_2d_pre, l_V_array_2d, radius, plotfigure_tag, vessel_removal=vessel_removal)
                                        liver_without_vessel_V_win = array_proc.crop_single_phase_single_img(t_V_array_2d_pre, j, liver_without_vessel_V, args.element_size)
                                        liver_without_vessel_V_win_remove_large = array_proc.crop_single_phase_single_img(t_V_array_2d_pre, j, liver_without_vessel_V_remove_large, args.element_size)
                                        #------------------------------------------------------------------------------------

                                        # 2. Calculate Washout Score

                                        if auto_segmentation:
                                            tumor_OTSU_bright_V, around_pix_img_mean_value_V, t_point_img_mean_value_V, tumor_OTSU_dark_V, enhancement_area_ratio_V, whole_enhancement_V = APHE_Washout_ZhongShan(V_2d_win,
                                                                                                              V_t_label_win,
                                                                                                              V_l_array_win,
                                                                                                              j,
                                                                                                              args.around_dilate_size,
                                                                                                              radius, 'V',
                                                                                                              data_site,
                                                                                                              args.rescaled,
                                                                                                              'Washout',
                                                                                                              Whole_enhancement[
                                                                                                                  i])
                                        else:
                                            tumor_OTSU_bright_V, around_pix_img_mean_value_V, t_point_img_mean_value_V, tumor_OTSU_dark_V, enhancement_area_ratio_V, whole_enhancement_V = APHE_Washout_ZhongShan(V_2d_win,
                                                                                                              V_t_label_win,
                                                                                                              V_l_array_win,
                                                                                                              j,
                                                                                                              args.around_dilate_size,
                                                                                                              radius, 'V',
                                                                                                              data_site,
                                                                                                              args.rescaled,
                                                                                                              'Washout',
                                                                                                              A_whole_enhancement=False)

                                        if auto_segmentation and Whole_enhancement[i]:
                                            tumorV = tumor_OTSU_dark_V

                                            # tumorV = t_point_img_mean_value_V # For Comparison Experiment

                                        else:
                                            tumorV = t_point_img_mean_value_V

                                        liverV = around_pix_img_mean_value_V

                                        scoreV = tumorV/liverV
                                        scoreVs += [scoreV]
                                        tumorVs += [tumorV]
                                        liverVs += [liverV]

                                        scores_Washout.append(scoreV)

                                        print('scoreWashout, scoreV, tumorV, liverV')
                                        print(1 / scoreV, scoreV, tumorV, liverV)


                                        Enhancement_area_ratio_V += [enhancement_area_ratio_V]



                                        diameters.append(Lesion_details[len(Lesion_details) - 1][7])


                                    plt.close('all')

                        max_diameter_V = max(diameters)

                    if Calculate_Capsule:
                        print('###############################  For Capsule Based on Portal Venous Phase #################################################')
                        for i in range(0, num_layers):

                            if logging_tag:
                                print('layer num: ', i + 1)


                            layer_index = i


                            V_array_2d, t_V_array_2d_pre, l_V_array_2d = pre_processing_single_layer_single_phase_ZhongShan(
                                V_layers, t_V_layers, l_V_layers, layer_index, file)
                            t_array_2d_max = 1

                            if t_V_array_2d_pre.max() == t_V_array_2d_pre.min():

                                if write_result:
                                    scores_Washout.append(0)
                                    diameters.append(0)
                                    ################################# Calculate APHE Score ##############################################

                                    if i < len(tz_layers_A) and (Calculate_APHE or Calculate_Washout):
                                        Lesion_details, Lesion_layers = tumor_details(t_A_array_2d_pre, spacing, tz_layers_A, i,
                                                                                      Lesion_details, Lesion_layers)
                                print('No lesion larger than 1 in this layer !!')
                            else:

                                # %%
                                for j in range(1, t_array_2d_max + 1):
                                    # %%

                                    V_2d_win, V_t_label_win, V_l_array_win = array_proc.crop_single_phase_ZhongShan \
                                        (t_V_array_2d_pre, j, V_array_2d, l_V_array_2d, args.element_size)


                                    if write_result:
                                        ################################# Calculate APHE Washout and Capsule Score ##############################################

                                        Lesion_details, Lesion_layers = tumor_details(t_V_array_2d_pre, spacing, tz_layers_V, i,
                                                                                      Lesion_details, Lesion_layers)

                                        radius = Lesion_details[len(Lesion_details) - 1][6] * 0.5

                                        #---------------------------------------------------------------------------------
                                        liver_without_vessel_V, liver_without_vessel_V_remove_large = liver_vessel_removal(V_array_2d, t_V_array_2d_pre, l_V_array_2d, radius, plotfigure_tag, vessel_removal=vessel_removal)
                                        liver_without_vessel_V_win = array_proc.crop_single_phase_single_img(t_V_array_2d_pre, j, liver_without_vessel_V, args.element_size)
                                        liver_without_vessel_V_win_remove_large = array_proc.crop_single_phase_single_img(t_V_array_2d_pre, j, liver_without_vessel_V_remove_large, args.element_size)
                                        #------------------------------------------------------------------------------------


                                        # 3. Calculate Capsule Score

                                        parser_Capsule, args_Capsule, args_text_Capsule = capsule_parameters(data_site)



                                        capsule_score_V = Capsule_ZhongShan2(V_2d_win, V_t_label_win, V_l_array_win,
                                                                             liver_without_vessel_V_win, \
                                                                             liver_without_vessel_V_win_remove_large,
                                                                             args.padding_tag, args.frangi_thresholded_tag, \
                                                                             radius, args.inside_threshold,
                                                                             args.outside_threshold, args.dilate_k,
                                                                             args.frangi_threshold, \
                                                                             plotfigure_tag, phase='V',
                                                                             vessel_removal=vessel_removal,
                                                                             remove_tumor_dark=args.remove_tumor_dark, \
                                                                             liver_rim_erosion=args.liver_rim_erosion, id=file,
                                                                             layer=i, args=args_Capsule)




                                        print('capsule_score_V: ', capsule_score_V)


                                        capsule_score = capsule_score_V

                                        ################################# Calculate Capsule Score ##################################################

                                        scores_Capsule.append(capsule_score)



                                        if Not_All_Slices and capsule_score_V > Capsule_Score_Threshold:
                                            print('!!!!!!!!!!!!!!!!!!!!!!  Find Capsule in V, Exist for Capsule recognition in V. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                                            i = num_layers

                            if i == num_layers:
                                break
                        if write_result:

                            scores_Capsule = [x for x in scores_Capsule if math.isnan(x) == False]
                            if len(scores_Capsule) > 0:
                                grade_V = max(scores_Capsule)
                                # grade1 = max(scores1)

                                max_layer_V = np.argmax(scores_Capsule)
                                # max_layer1 = np.argmax(scores1)
                                # grade = scores
                            else:
                                grade_V = 0
                                max_layer_V = 0

                            print('############################################# Capsule Final Score based on V is: ', grade_V,
                                  ' ##############################')


                else:
                    print('####################################################  Need to Check   ############################')


                # Calculate based on phase D

                print(' ###########################  Calculate based on phase D ...  ##################################')
                num_layers = tz_layers_D.shape[0]
                # num_layers = len(layers)
                print('Total layers D: ', num_layers)
                # %%
                if num_layers > 0:
                    if write_result:
                        scores_Washout = []
                        scores_Capsule = []
                        scoreDs = []

                        diameters_Washout = []
                        diameters_Capsule = []

                        Lesion_details = []
                        Lesion_layers = []

                    if Calculate_Capsule and (not Not_All_Slices or grade_V <= Capsule_Score_Threshold):
                        print('############################### Calculate Capsule Score based on Delay ################################')
                        for i in range(0, num_layers):

                            if logging_tag:
                                print('layer num: ', i + 1)

                            layer_index = i


                            D_array_2d, t_D_array_2d_pre, l_D_array_2d = pre_processing_single_layer_single_phase_ZhongShan(
                                D_layers, t_D_layers, l_D_layers, layer_index, file)
                            t_array_2d_max = 1

                            if t_D_array_2d_pre.max() == t_D_array_2d_pre.min():

                                if write_result:
                                    scores_Washout.append(0)
                                    diameters_Washout.append(0)

                                    scores_Capsule.append(0)
                                    diameters_Capsule.append(0)

                                    ################################# Calculate APHE Score ##############################################

                                    Lesion_details, Lesion_layers = tumor_details(t_D_array_2d_pre, spacing, tz_layers_D, i,
                                                                                  Lesion_details, Lesion_layers)
                                print('No lesion larger than 1 in this layer !!')
                            else:

                                # %%
                                for j in range(1, t_array_2d_max + 1):
                                    # %%

                                    D_2d_win, D_t_label_win, D_l_array_win = array_proc.crop_single_phase_ZhongShan \
                                        (t_D_array_2d_pre, j, D_array_2d, l_D_array_2d, args.element_size)


                                    if write_result:
                                        ################################# Calculate APHE Washout and Capsule Score ##############################################

                                        Lesion_details, Lesion_layers = tumor_details(t_D_array_2d_pre, spacing,
                                                                                      tz_layers_D, i,
                                                                                      Lesion_details, Lesion_layers)

                                        radius = Lesion_details[len(Lesion_details) - 1][6] * 0.5

                                        # ---------------------------------------------------------------------------------
                                        liver_without_vessel_D, liver_without_vessel_D_remove_large = liver_vessel_removal(
                                            D_array_2d, t_D_array_2d_pre, l_D_array_2d, radius, plotfigure_tag,
                                            vessel_removal=vessel_removal)
                                        liver_without_vessel_D_win = array_proc.crop_single_phase_single_img(
                                            t_D_array_2d_pre, j,
                                            liver_without_vessel_D,
                                            args.element_size)
                                        liver_without_vessel_D_win_remove_large = array_proc.crop_single_phase_single_img(
                                            t_D_array_2d_pre, j, liver_without_vessel_D_remove_large, args.element_size)


                                        # 3. Calculate Capsule Score
                                        capsule_score_D = Capsule_ZhongShan2(D_2d_win, D_t_label_win, D_l_array_win,
                                                                             liver_without_vessel_V_win, \
                                                                             liver_without_vessel_V_win_remove_large,
                                                                             args.padding_tag,
                                                                             args.frangi_thresholded_tag, radius,
                                                                             args.inside_threshold,
                                                                             args.outside_threshold, args.dilate_k, args.frangi_threshold,
                                                                             plotfigure_tag, phase='D',
                                                                             vessel_removal=vessel_removal,
                                                                             remove_tumor_dark=args.remove_tumor_dark,
                                                                             liver_rim_erosion=args.liver_rim_erosion, id=file, layer=i, args=args_Capsule)

                                        print('capsule_score_D: ', capsule_score_D)

                                        capsule_score = capsule_score_D
                                        # capsule_score = 0

                                        ################################# Calculate Capsule Score ##################################################

                                        # scores.append(aphe_score)

                                        # scores.append(washout_score)
                                        scores_Capsule.append(capsule_score)

                                        # result = [tumorPre, liverPre, tumorA, liverA, tumorV, liverV]
                                        # scores.append(result)

                                        diameters_Capsule.append(Lesion_details[len(Lesion_details) - 1][7])

                                        if Not_All_Slices and capsule_score_D > Capsule_Score_Threshold:
                                            print('!!!!!!!!!!!!!!!!!!!!!!  Find Capsule in D, Exist for Capsule recognition in D. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                                            i = num_layers

                            if i == num_layers:
                                break

                        if write_result:
                            print('Write result for Capsule')
                            scores_Capsule = [x for x in scores_Capsule if math.isnan(x) == False]
                            grade_D = max(scores_Capsule)
                            # grade1 = max(scores1)

                            max_layer_D = np.argmax(scores_Capsule)
                            # max_layer1 = np.argmax(scores1)
                            # grade = scores
                            max_diameter_D = max(diameters_Capsule)
                            print('############################################# Capsule Final Score based on Delay is: ', grade_D,
                                  ' ##############################')

                else:
                    print(
                        '####################################################  Need to Check   ############################')


                if Calculate_Washout:
                    print('######################################### Calculate Washout Final Score ##################################')
                    grade_Washout, max_layer_Washout = Washout_Score(data_site, scoreVs, scoreDs, liverVs, tumorVs, ratioTumorAtoPre, ratioScoreAtoPre, tumorAs, liverAs, scoreAs, V_bright_around_all, V_bright_area_ratio_A_all, V_bright_area_ratio_A_threshold, \
                      TumorVBrightMeans, TumorABrightMeans, V_TumorBright_Liver_intensity_ratio_threshold, use_manual_parameters=True, args=args, auto_segmentation=auto_segmentation, WashInFirst=WashInFirst)



                    print('################################# Washout Final Score is: ', grade_Washout, "################################")

                    print('Write for Washout')
                    csv_writer.write_to_csv(csvfile_Washout, ['final_score:', grade_Washout])

                    csvfile_Washout.close()


                if Calculate_Capsule:
                    print('Write for Capsule')
                    if (Not_All_Slices and grade_V > Capsule_Score_Threshold) or ((not Not_All_Slices or grade_V <= Capsule_Score_Threshold) and grade_V >= grade_D):

                        grade = grade_V
                        max_layer = max_layer_V
                        max_layer_ = tz_layers_V[max_layer_V]
                        score_max_phase = 'V'
                    else:
                        grade = grade_D
                        max_layer = max_layer_D
                        max_layer_ = tz_layers_D[max_layer_D]
                        # print(tz_layers_D)
                        # print(max_layer_D)
                        score_max_phase = 'D'

                    if Calculate_Washout and grade_V <= Capsule_Score_Threshold:
                        if max_diameter_V >= max_diameter_D:
                            max_diameter = max_diameter_V
                            diameter_max_phase = 'V'
                        else:
                            max_diameter = max_diameter_D
                            diameter_max_phase = 'D'

                        csv_writer.write_to_csv(csvfile_Capsule, ['final_score:' + str(grade), ' at phase:' + score_max_phase, ' at layer:' + str(max_layer_+1), ' max diameter:' + str(max_diameter), ' at phase:' + diameter_max_phase])
                    else:
                        csv_writer.write_to_csv(csvfile_Capsule,
                                                ['final_score:' + str(grade), ' at phase:' + score_max_phase,
                                                 ' at layer:' + str(max_layer_ + 1)])
                    csvfile_Capsule.close()

                print('Done!')


                if Calculate_Washout:
                    csv_writer.write_to_csv(csvfile_all_Washout, [file, grade_Washout, max_layer_Washout])

                if Calculate_Capsule:
                    if Calculate_Washout:
                        csv_writer.write_to_csv(csvfile_all_Capsule, [file, grade, max_layer, max_diameter])
                    else:
                        csv_writer.write_to_csv(csvfile_all_Capsule, [file, grade, max_layer])

                end_time_each = time.time()

                time_each = end_time_each - start_time_each
                csv_writer.write_to_csv(csvfile_all_Time, [file, time_each])


            if write_result:
                csvfile_all_Time.close()

                if Calculate_APHE:
                    csvfile_all_APHE.close()
                    with open(os.path.join('./result/APHE/' + result_dir + '/', 'LIRADS_main_args.yaml'), 'w') as f:
                        f.write(args_text)
                if Calculate_Washout:
                    csvfile_all_Washout.close()
                    with open(os.path.join('./result/Washout/' + result_dir + '/', 'LIRADS_main_args.yaml'), 'w') as f:
                        f.write(args_text)
                if Calculate_Capsule:
                    csvfile_all_Capsule.close()
                    with open(os.path.join('./result/Capsule/' + result_dir + '/', 'LIRADS_main_args.yaml'), 'w') as f:
                        f.write(args_text)

                    with open(os.path.join('./result/Capsule/' + result_dir + '/', 'Capsule_args.yaml'), 'w') as f:
                        f.write(args_text_Capsule)


            print('Totally Done!')



cProfile.run('main()')
