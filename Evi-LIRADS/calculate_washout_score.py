import numpy as np
import math

def Washout_Score(data_site, scoreVs, scoreDs, liverVs, tumorVs, ratioTumorAtoPre, ratioScoreAtoPre, tumorAs, liverAs, scoreAs, V_bright_around_all, V_bright_area_ratio_A_all, V_bright_area_ratio_A_threshold, \
                  TumorVBrightMeans, TumorABrightMeans, V_TumorBright_Liver_intensity_ratio_threshold, use_manual_parameters, args, auto_segmentation, WashInFirst):

    if data_site == 'ZheYi':
        scoreVs = [x for x in scoreVs if math.isnan(x) == False]
        scoreDs = [x for x in scoreDs if math.isnan(x) == False]
        scoreVD = min(min(scoreVs), min(scoreDs))
        max_layer_V = np.argmin(scoreVs)
        max_layer_D = np.argmin(scoreDs)
        if min(scoreVs) <= min(scoreDs):
            max_layer = max_layer_V
        else:
            max_layer = max_layer_D

        washout_score = 1 / scoreVD

    if data_site == 'ZhongShan' or data_site == 'SuZhou' or 'PHC':

        print('liverVs All: ', liverVs)
        print('tumorVs All: ', tumorVs)
        print('scoreVs All: ', scoreVs)

        if not auto_segmentation:
            # scoreVs = [x for x in scoreVs if math.isnan(x) == False]
            tumorVs = [x for x in tumorVs if math.isnan(x) == False]
            # liverVs = [x for x in liverVs if math.isnan(x) == False]

            for lV in range(0, len(liverVs)):
                if math.isnan(liverVs[lV]) == True:
                    liverVs[lV] = 10000000

            for sV in range(0, len(scoreVs)):
                if math.isnan(scoreVs[sV]) == True:
                    scoreVs[sV] = 1 / 10000000
        else:
            # -------------------------------- #
            kept_index = []
            for lV in range(0, len(liverVs)):
                if math.isnan(liverVs[lV]) == False:
                    kept_index.append(lV)

            liverVs = [liverVs[p] for p in kept_index]
            scoreVs = [scoreVs[p] for p in kept_index]
            tumorVs = [tumorVs[p] for p in kept_index]
            # -------------------------------- #

        if auto_segmentation and WashInFirst:
            # judge if wash in exists
            print('ratioTumorAtoPre')
            print(ratioTumorAtoPre)
            print('ratioScoreAtoPre')
            print(ratioScoreAtoPre)

            print('scoreAs')
            print(scoreAs)
            # print('tumorAs')
            # print(tumorAs)
            # print('tumorVs')
            # print(tumorVs)

            kept_index = []
            for lA in range(0, len(scoreAs)):
                if math.isnan(scoreAs[lA]) == False:
                    kept_index.append(lA)

            scoreAs = [scoreAs[p] for p in kept_index]
            ratioScoreAtoPre = [ratioScoreAtoPre[p] for p in kept_index]
            ratioTumorAtoPre = [ratioTumorAtoPre[p] for p in kept_index]

            V_bright_around_all = [x for x in V_bright_around_all if math.isnan(x) == False]
            V_bright_area_ratio_A_all = [x for x in V_bright_area_ratio_A_all if math.isnan(x) == False]

            if args.consider_pre and ((args.WashoutCompareTumorAtoTumorPre and (
                    not (min(ratioTumorAtoPre) > 1 and max(ratioScoreAtoPre) > 1 and max(
                        scoreAs) > args.scoreAThreshold))) or max(scoreAs) <= args.scoreAThreshold):
                washout_score = 1 / 10
                max_layer = 0

            # elif max(V_bright_around_all) > V_TumorBright_Liver_intensity_ratio_threshold:
            #     print('V：Tumor较亮区域信号强度比Liver大： ', max(V_bright_around_all))
            #     # scoreVD = 100
            #     washout_score = 0.01
            #     max_layer = 0

            elif max(V_bright_area_ratio_A_all) > V_bright_area_ratio_A_threshold:
                print('渐进性强化, 最大面积占比： ', max(V_bright_area_ratio_A_all))
                # scoreVD = 50
                washout_score = 0.02
                max_layer = 0
            # if False:
            #     print('Never into this loop')
            else:
                scoreVs_ = []
                for tumorV_index in range(0, len(tumorVs)):

                    tumorV_ = tumorVs[tumorV_index]
                    liverV_ = liverVs[tumorV_index]
                    # adjacentA = []

                    if tumorV_index >= 0 and tumorV_index <= len(tumorAs) - 1:
                        # adjacentA.append(tumorAs[tumorV_index])
                        tumorA_ = tumorAs[tumorV_index]
                        liverA_ = liverAs[tumorV_index]
                    elif tumorV_index - 1 >= 0 and tumorV_index - 1 <= len(tumorAs) - 1:
                        # adjacentA.append(tumorAs[tumorV_index - 1])
                        tumorA_ = tumorAs[tumorV_index - 1]
                        liverA_ = liverAs[tumorV_index - 1]
                    elif tumorV_index + 1 >= 0 and tumorV_index + 1 <= len(tumorAs) - 1:
                        # adjacentA.append(tumorAs[tumorV_index + 1])
                        tumorA_ = tumorAs[tumorV_index + 1]
                        liverA_ = liverAs[tumorV_index + 1]
                    else:
                        tumorA_ = 0
                        liverA_ = 0

                    # if len(adjacentA) > 0:
                    #     if tumorV_ < np.max(adjacentA):

                    if use_manual_parameters:
                        VA_condition = tumorA_ > 0 and tumorV_ / liverV_ < tumorA_ / liverA_
                        # and Enhancement_area_ratio_V[tumorV_index] < 1.1 * Enhancement_area_ratio_A[
                        #             tumorV_index]

                    else:
                        VA_condition = (tumorA_ > 0) and (
                                tumorV_ / liverV_ < 0.95 * tumorA_ / liverA_) \
                                       and ((
                                TumorVBrightMeans[tumorV_index] < 1.3 * TumorABrightMeans[
                            tumorV_index]))

                        #         \
                        #         or (TumorVDarkMeans[tumorV_index] < 1.2*TumorADarkMeans[tumorV_index])):  # V和A比较
                        # if (tumorA_ > 0) and ((TumorVBrightMeans[tumorV_index] < 1.1 * TumorABrightMeans[tumorV_index]) \
                        #         or (TumorVDarkMeans[tumorV_index] < 1.1 * TumorADarkMeans[tumorV_index])):  # V和A比较

                    if VA_condition:
                        # print('tumorV_index', 'tumorV_', 'tumorA_', 'tumorV_ / liverV_', 'tumorA_ / liverA_')
                        # print(tumorV_index, tumorV_, tumorA_, tumorV_/liverV_, tumorA_/liverA_)
                        scoreVs_.append(scoreVs[tumorV_index])
                    # elif tumorA_ == 0:
                    #     scoreVs_.append(scoreVs[tumorV_index])

                print('scoreVs_')
                print(scoreVs_)

                if args.compareTumorVwithTumorA_normalizedByLiver:
                    if len(scoreVs_) > 0:
                        scoreVD = min(scoreVs_)
                        max_layer = np.argmin(scoreVs_)
                    else:
                        scoreVD = 20
                        max_layer = 0
                else:
                    scoreVD = min(scoreVs)
                    max_layer = np.argmin(scoreVs)

                washout_score = 1 / scoreVD
        elif auto_segmentation:
            scoreVs_ = []
            for tumorV_index in range(0, len(tumorVs)):

                tumorV_ = tumorVs[tumorV_index]
                liverV_ = liverVs[tumorV_index]
                # adjacentA = []

                if tumorV_index >= 0 and tumorV_index <= len(tumorAs) - 1:
                    # adjacentA.append(tumorAs[tumorV_index])
                    tumorA_ = tumorAs[tumorV_index]
                    liverA_ = liverAs[tumorV_index]
                elif tumorV_index - 1 >= 0 and tumorV_index - 1 <= len(tumorAs) - 1:
                    # adjacentA.append(tumorAs[tumorV_index - 1])
                    tumorA_ = tumorAs[tumorV_index - 1]
                    liverA_ = liverAs[tumorV_index - 1]
                elif tumorV_index + 1 >= 0 and tumorV_index + 1 <= len(tumorAs) - 1:
                    # adjacentA.append(tumorAs[tumorV_index + 1])
                    tumorA_ = tumorAs[tumorV_index + 1]
                    liverA_ = liverAs[tumorV_index + 1]
                else:
                    tumorA_ = 0
                    liverA_ = 0

                # if len(adjacentA) > 0:
                #     if tumorV_ < np.max(adjacentA):

                if use_manual_parameters:
                    VA_condition = tumorA_ > 0 and tumorV_ / liverV_ < tumorA_ / liverA_
                    # and Enhancement_area_ratio_V[tumorV_index] < 1.1 * Enhancement_area_ratio_A[
                    #             tumorV_index]

                else:
                    VA_condition = (tumorA_ > 0) and (tumorV_ / liverV_ < 0.95 * tumorA_ / liverA_) \
                                   and ((TumorVBrightMeans[tumorV_index] < 1.3 * TumorABrightMeans[tumorV_index]))

                    #         \
                    #         or (TumorVDarkMeans[tumorV_index] < 1.2*TumorADarkMeans[tumorV_index])):  # V和A比较
                    # if (tumorA_ > 0) and ((TumorVBrightMeans[tumorV_index] < 1.1 * TumorABrightMeans[tumorV_index]) \
                    #         or (TumorVDarkMeans[tumorV_index] < 1.1 * TumorADarkMeans[tumorV_index])):  # V和A比较

                if VA_condition:
                    # print('tumorV_index', 'tumorV_', 'tumorA_', 'tumorV_ / liverV_', 'tumorA_ / liverA_')
                    # print(tumorV_index, tumorV_, tumorA_, tumorV_/liverV_, tumorA_/liverA_)
                    scoreVs_.append(scoreVs[tumorV_index])
                # elif tumorA_ == 0:
                #     scoreVs_.append(scoreVs[tumorV_index])

            V_bright_around_all = [x for x in V_bright_around_all if math.isnan(x) == False]
            V_bright_area_ratio_A_all = [x for x in V_bright_area_ratio_A_all if math.isnan(x) == False]

            if max(V_bright_around_all) > V_TumorBright_Liver_intensity_ratio_threshold:
                print('V：Tumor较亮区域信号强度比Liver大： ', max(V_bright_around_all))
                scoreVD = 100
                max_layer = 0
                # washout_score = 1 / scoreVD
            elif max(V_bright_area_ratio_A_all) > V_bright_area_ratio_A_threshold:
                print('渐进性强化, 最大面积占比： ', max(V_bright_area_ratio_A_all))
                scoreVD = 50
                max_layer = 0
                # washout_score = 1 / scoreVD
            else:
                if args.compareTumorVwithTumorA_normalizedByLiver:
                    if len(scoreVs_) > 0:
                        scoreVD = min(scoreVs_)
                        max_layer = np.argmin(scoreVs_)
                    else:
                        scoreVD = 20
                        max_layer = 0
                else:
                    scoreVD = min(scoreVs)
                    max_layer = np.argmin(scoreVs)

            # judge if wash in exists
            print('ratioTumorAtoPre')
            print(ratioTumorAtoPre)
            print('ratioScoreAtoPre')
            print(ratioScoreAtoPre)
            print('scoreAs')
            print(scoreAs)
            print('tumorAs')
            print(tumorAs)
            print('tumorVs')
            print(tumorVs)
            print('scoreVs_')
            print(scoreVs_)

            kept_index = []
            for lA in range(0, len(scoreAs)):
                if math.isnan(scoreAs[lA]) == False:
                    kept_index.append(lA)

            scoreAs = [scoreAs[p] for p in kept_index]
            ratioScoreAtoPre = [ratioScoreAtoPre[p] for p in kept_index]
            ratioTumorAtoPre = [ratioTumorAtoPre[p] for p in kept_index]

            if args.consider_pre:
                if args.WashoutCompareTumorAtoTumorPre:
                    if min(ratioTumorAtoPre) > 1 and max(ratioScoreAtoPre) > 1 and max(scoreAs) > args.scoreAThreshold:
                        washout_score = 1 / scoreVD
                    else:
                        washout_score = 1 / 10
                        max_layer = 0
                else:
                    if max(scoreAs) > args.scoreAThreshold:
                        washout_score = 1 / scoreVD
                    else:
                        washout_score = 1 / 10
                        max_layer = 0
            else:
                washout_score = 1 / scoreVD


        else:

            scoreVs_ = []
            for tumorV_index in range(0, len(tumorVs)):

                tumorV_ = tumorVs[tumorV_index]
                liverV_ = liverVs[tumorV_index]
                # adjacentA = []

                if tumorV_index >= 0 and tumorV_index <= len(tumorAs) - 1:
                    # adjacentA.append(tumorAs[tumorV_index])
                    tumorA_ = tumorAs[tumorV_index]
                    liverA_ = liverAs[tumorV_index]
                elif tumorV_index - 1 >= 0 and tumorV_index - 1 <= len(tumorAs) - 1:
                    # adjacentA.append(tumorAs[tumorV_index - 1])
                    tumorA_ = tumorAs[tumorV_index - 1]
                    liverA_ = liverAs[tumorV_index - 1]
                elif tumorV_index + 1 >= 0 and tumorV_index + 1 <= len(tumorAs) - 1:
                    # adjacentA.append(tumorAs[tumorV_index + 1])
                    tumorA_ = tumorAs[tumorV_index + 1]
                    liverA_ = liverAs[tumorV_index + 1]
                else:
                    tumorA_ = 0
                    liverA_ = 0

                # if len(adjacentA) > 0:
                #     if tumorV_ < np.max(adjacentA):

                VA_condition = tumorA_ > 0 and tumorV_ / liverV_ < tumorA_ / liverA_  # V和A比较

                if VA_condition:
                    # print('tumorV_index', 'tumorV_', 'tumorA_', 'tumorV_ / liverV_', 'tumorA_ / liverA_')
                    # print(tumorV_index, tumorV_, tumorA_, tumorV_/liverV_, tumorA_/liverA_)
                    scoreVs_.append(scoreVs[tumorV_index])
                elif tumorA_ == 0:
                    scoreVs_.append(scoreVs[tumorV_index])

            if args.compareTumorVwithTumorA_normalizedByLiver:
                if len(scoreVs_) > 0:
                    scoreVD = min(scoreVs_)
                    max_layer = np.argmin(scoreVs_)
                else:
                    scoreVD = 20
                    max_layer = 0
            else:
                scoreVD = min(scoreVs)
                max_layer = np.argmin(scoreVs)

            # judge if wash in exists
            print('ratioTumorAtoPre')
            print(ratioTumorAtoPre)
            print('ratioScoreAtoPre')
            print(ratioScoreAtoPre)
            print('scoreAs')
            print(scoreAs)
            print('tumorAs')
            print(tumorAs)
            print('tumorVs')
            print(tumorVs)
            print('scoreVs_')
            print(scoreVs_)

            ############  -----------------注意：当列表中有nan时，求max的结果是nan；因此，要先移除列表中的nan；之前基于手动mask没有移除nan，结果是有点问题的--------------- ###########################
            if auto_segmentation:
                kept_index = []
                for lA in range(0, len(scoreAs)):
                    if math.isnan(scoreAs[lA]) == False:
                        kept_index.append(lA)

                scoreAs = [scoreAs[p] for p in kept_index]
                ratioScoreAtoPre = [ratioScoreAtoPre[p] for p in kept_index]
                ratioTumorAtoPre = [ratioTumorAtoPre[p] for p in kept_index]
            ############  ------------------------------------------------------------------------------------------------------------------------------  ###########################

            if args.consider_pre:
                if args.WashoutCompareTumorAtoTumorPre:
                    if max(ratioTumorAtoPre) > 1 and max(scoreAs) > args.scoreAThreshold:  # and max(ratioScoreAtoPre) > 1
                        washout_score = 1 / scoreVD
                    else:
                        washout_score = 1 / 10
                        max_layer = 0
                else:
                    if max(scoreAs) > args.scoreAThreshold:
                        washout_score = 1 / scoreVD
                    else:
                        washout_score = 1 / 10
                        max_layer = 0
            else:
                washout_score = 1 / scoreVD

    return washout_score, max_layer