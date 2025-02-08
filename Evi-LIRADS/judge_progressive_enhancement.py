#########################################################################################################
# 2-2.  -----------------------  判断TumorV是否比TumorA信号强度低，用V较亮区域和A较亮区域比  （判断是否是渐进性强化）  ---------------------------
#########################################################################################################

import copy
import numpy as np
from matplotlib import pyplot as plt
from Enhancement import Tumor_Backaround

def if_progressive_enhancement(A_2d_win, A_t_label_win, A_l_array_win, V_2d_win, V_t_label_win, V_l_array_win, radius, liver_enhancement_k, whole_enhancement, auto_segmentation, plotfigure_tag):
    # For Washout
    # print('For Washout: ')
    V_bright_area_ratio_A = 0 # 判断是否是渐进性强化
    ###################  A   ###########################
    # A_2d_win, A_t_label_win, A_l_array_win
    A_2d_win_tumor = copy.deepcopy(A_2d_win)
    y_t, x_t = np.where(A_t_label_win == 0)[0], np.where(A_t_label_win == 0)[1]
    A_2d_win_tumor[y_t, x_t] = 0

    # 将二维像素值按照信号强度排序，并将信号值较小的2%置0，去暗区域
    # 将二维数组展平，并去除所有的0值
    flattened_nonzero = A_2d_win_tumor[A_2d_win_tumor != 0].flatten()

    A_2d_win_bright = copy.deepcopy(A_2d_win_tumor)
    A_2d_win_dark = copy.deepcopy(A_2d_win_tumor)

    # 如果去除0后数组为空，则直接返回原数组
    if flattened_nonzero.size != 0:

        if plotfigure_tag:
            plt.figure()
            plt.subplot(2, 4, 1)
            plt.imshow(A_2d_win, cmap='gray')
            plt.title('A_2d_win')

            plt.subplot(2, 4, 2)
            plt.imshow(A_2d_win_tumor, cmap='gray')
            plt.title('A_2d_win_tumor')


        # 计算需要置0的阈值，即去除0值后顶部5%的信号强度
        threshold = np.percentile(flattened_nonzero,
                                  75)

        # 将所有大于或等于阈值的元素置为0
        A_2d_win_bright[A_2d_win_bright <= threshold] = 0
        A_2d_win_dark[A_2d_win_dark > threshold] = 0

        A_2d_win_bright_mean = np.mean(A_2d_win_bright[A_2d_win_bright != 0])
        A_2d_win_dark_mean = np.mean(A_2d_win_dark[A_2d_win_dark != 0])

        exist_Tumor_Bright_A = (A_2d_win_bright != 0)
        A_bright_area = exist_Tumor_Bright_A.sum()



            # plt.subplot(2, 4, 4)
            # plt.imshow(A_2d_win_dark, cmap='gray')
            # plt.title('A_2d_win_dark')
            #
            # plt.text(5, 5,
            #          'mean of TumorADark: ' + str(
            #              round(A_2d_win_dark_mean, 2)),
            #          fontdict=dict(fontsize=12, color='r', family='monospace',
            #                        weight='bold'))

            # plt.show()
    ###################  End A  ###############################

    ####################  V   #################################
    if auto_segmentation:
        V_2d_win_tumor = copy.deepcopy(V_2d_win)
        y_t, x_t = np.where(V_t_label_win == 0)[0], np.where(V_t_label_win == 0)[1]
        V_2d_win_tumor[y_t, x_t] = 0

        # 将二维像素值按照信号强度排序，并将信号值较小的2%置0，去暗区域
        # 将二维数组展平，并去除所有的0值
        flattened_nonzero = V_2d_win_tumor[V_2d_win_tumor != 0].flatten()

        V_2d_win_bright = copy.deepcopy(V_2d_win_tumor)
        V_2d_win_dark = copy.deepcopy(V_2d_win_tumor)

        # 如果去除0后数组为空，则直接返回原数组
        if flattened_nonzero.size != 0:

            if plotfigure_tag:
                plt.subplot(2, 4, 5)
                plt.imshow(V_2d_win, cmap='gray')
                plt.title('V_2d_win')
                plt.subplot(2, 4, 6)
                plt.imshow(V_2d_win_tumor, cmap='gray')
                plt.title('V_2d_win_tumor')


            # 计算需要置0的阈值，即去除0值后顶部5%的信号强度
            threshold = np.percentile(flattened_nonzero,
                                      75)

            # 将所有大于或等于阈值的元素置为0
            V_2d_win_bright[V_2d_win_bright <= threshold] = 0
            V_2d_win_dark[V_2d_win_dark > threshold] = 0

            V_2d_win_bright_mean = np.mean(V_2d_win_bright[V_2d_win_bright != 0])
            V_2d_win_dark_mean = np.mean(V_2d_win_dark[V_2d_win_dark != 0])

            exist_Tumor_Bright_V = (V_2d_win_bright != 0)
            V_bright_area = exist_Tumor_Bright_V.sum()

            exist_Tumor_V = (V_2d_win_tumor != 0)
            V_area = exist_Tumor_V.sum()

            V_bright_area_ratio = V_bright_area/V_area #


            # 计算TumorVBright/LiverV, TumorABright/LiverA
            around_pix_img_A = Tumor_Backaround(A_2d_win, A_t_label_win, A_l_array_win, 1,
                             radius, 'A', 'Washout')
            around_pix_img_V = Tumor_Backaround(V_2d_win, V_t_label_win,
                                                V_l_array_win, 1,
                                                radius, 'V', 'Washout')

            A_around_mean = np.mean(around_pix_img_A[around_pix_img_A != 0])
            V_around_mean = np.mean(around_pix_img_V[around_pix_img_V != 0])

            V_bright_around = V_2d_win_bright_mean / V_around_mean #

            if plotfigure_tag:
                plt.subplot(2, 4, 3)
                plt.imshow(A_2d_win_bright, cmap='gray')
                plt.title('A_2d_win_bright')

                plt.text(5, 5,
                         'mean of TumorABright: ' + str(
                             round(A_2d_win_bright_mean, 2)),
                         fontdict=dict(fontsize=12, color='r',
                                       family='monospace',
                                       weight='bold'))

                plt.text(5, 15,
                         'TumorABright / A Around: ' + str(
                             round(A_2d_win_bright_mean / A_around_mean,
                                   2)),
                         fontdict=dict(fontsize=12, color='r',
                                       family='monospace',
                                       weight='bold'))

                # plt.subplot(2, 4, 4)
                # plt.imshow(around_pix_img_A, cmap='gray')
                # plt.title('around_pix_img_A')
                #
                # plt.text(5, 5,
                #          'mean of TumorAAround: ' + str(
                #              round(A_around_mean, 2)),
                #          fontdict=dict(fontsize=12, color='r',
                #                        family='monospace',
                #                        weight='bold'))

                plt.subplot(2, 4, 7)
                plt.imshow(V_2d_win_bright, cmap='gray')
                plt.title('V_2d_win_bright')

                plt.text(5, 5,
                         'mean of TumorVBright: ' + str(
                             round(V_2d_win_bright_mean, 2)),
                         fontdict=dict(fontsize=12, color='r',
                                       family='monospace',
                                       weight='bold'))

                plt.text(5, 15,
                         'TumorVBright / V Around: ' + str(
                             round(V_2d_win_bright_mean / V_around_mean,
                                   2)),
                         fontdict=dict(fontsize=12, color='r',
                                       family='monospace',
                                       weight='bold'))

                # plt.text(5, 25,
                #          'TumorVBright Area Ratio: ' + str(
                #              round(
                #                  V_bright_area_ratio,
                #                  2)),
                #          fontdict=dict(fontsize=12, color='r',
                #                        family='monospace',
                #                        weight='bold'))

                # plt.text(5, 30,
                #          'TumorVBright/TumorABright: ' + str(
                #              round(
                #                  V_2d_win_bright_mean / A_2d_win_bright_mean,
                #                  2)),
                #          fontdict=dict(fontsize=12, color='r',
                #                        family='monospace',
                #                        weight='bold'))




                # plt.subplot(2, 4, 8)
                # plt.imshow(around_pix_img_V, cmap='gray')
                # plt.title('around_pix_img_V')
                #
                # plt.text(5, 15,
                #          'mean of TumorVAround: ' + str(
                #              round(V_around_mean, 2)),
                #          fontdict=dict(fontsize=12, color='r',
                #                        family='monospace',
                #                        weight='bold'))

            if V_2d_win_bright_mean > A_2d_win_bright_mean and V_2d_win_bright_mean/V_around_mean > A_2d_win_bright_mean/(A_around_mean*liver_enhancement_k) and (not whole_enhancement):
                # A没有强化，V强化
                y,x = np.where(A_2d_win_bright > 0)[0], np.where(A_2d_win_bright > 0)[1]
                V_2d_win_bright_A = copy.deepcopy(V_2d_win_bright)
                V_2d_win_bright_A[y,x] = 0

                V_2d_win_bright_A[V_2d_win_bright_A <= V_around_mean] = 0

                exist_Tumor_Bright_V_A = (V_2d_win_bright_A != 0)
                V_bright_area_A = exist_Tumor_Bright_V_A.sum()

                exist_V = (V_2d_win != 0)
                V_area = exist_V.sum()

                V_bright_area_ratio_A = V_bright_area_A/V_area

                print('A没有强化，V强化: ', V_bright_area_ratio_A)

                if plotfigure_tag:

                    # plt.subplot(2, 4, 8)
                    # plt.imshow(V_2d_win_dark, cmap='gray')
                    # plt.title('V_2d_win_dark')
                    #
                    # plt.text(5, 5,
                    #          'mean of TumorVDark: ' + str(
                    #              round(V_2d_win_dark_mean, 2)),
                    #          fontdict=dict(fontsize=12, color='r', family='monospace',
                    #                        weight='bold'))
                    #
                    # plt.text(5, 15,
                    #          'TumorVDark/TumorADark: ' + str(
                    #              round(V_2d_win_dark_mean / A_2d_win_dark_mean, 2)),
                    #          fontdict=dict(fontsize=12, color='r', family='monospace',
                    #                        weight='bold'))


                    plt.subplot(2, 4, 8)
                    plt.imshow(V_2d_win_bright_A, cmap='gray')
                    plt.title('V_2d_win_bright_A')

                    # plt.text(5, 25,
                    #          'V_bright_area_A: ' + str(
                    #              round(V_bright_area_A, 2)),
                    #          fontdict=dict(fontsize=12, color='r', family='monospace',
                    #                        weight='bold'))

                    plt.text(5, 5,
                             'V_bright_area_ratio_A: ' + str(
                                 round(V_bright_area_ratio_A, 2)),
                             fontdict=dict(fontsize=12, color='r',
                                           family='monospace',
                                           weight='bold'))


            if plotfigure_tag:
                plt.show()

    return V_bright_area_ratio_A, V_2d_win_bright_mean, A_2d_win_bright_mean, V_2d_win_dark_mean, A_2d_win_dark_mean, V_bright_around