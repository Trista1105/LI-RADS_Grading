# -*- coding: utf-8 -*-
import os
import logging
import radiomics
from radiomics import featureextractor
from radiomics import logger as logger
from multiprocessing import Process, Pool
import SimpleITK as sitk

import numpy as np

import shutil
import csv
import copy


def radiomics_process(imageName,maskName,label_num):
    label_num=int(label_num)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    # Alternative: use hardcoded settings (separate for settings, input image types and enabled features)
    settings = {}
    # settings['binWidth'] = 25 #25, 60
    settings['binCount'] = 30
    settings['geometryTolerance'] = 1300 # original:1000
    settings['resampledPixelSpacing'] = None
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True
    settings['normalize'] = True

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
#    extractor.enableImageTypeByName('Wavelet')
    featureVector = extractor.execute(imageName, maskName,label=label_num)
    return featureVector

def xls_write_all(result_list, xlspath):
    fout1 = open(xlspath, 'w', newline='' '')
    csv_writer = csv.writer(fout1)
    # fout1.write("".join(result_list[0][0]) + '\n')
    csv_writer.writerow(result_list[0][0])
    for i in range(0, len(result_list)):
        feature_list = result_list[i][1]
        for j in range(0,len(feature_list)):
            # fout1.write("".join(feature_list[j]) + '\n')
            csv_writer.writerow(feature_list[j])
    fout1.close()

def xls_write_4phases(result_list, xlspath, organ):

    for phase in range(1,5):
        fout1 = open(xlspath + '/' + str(organ) + '_phase'+str(phase)+'.csv', 'w', newline='' '')
        csv_writer = csv.writer(fout1)
        # fout1.write("".join(result_list[0][0]) + '\n')
        csv_writer.writerow(result_list[0][0])
        for i in range(0, len(result_list)):
            feature_list = result_list[i][1]
            for j in range(0, len(feature_list)):
                if j%4 == phase-1:
                    # fout1.write("".join(feature_list[j]) + '\n')
                    csv_writer.writerow(feature_list[j])

        fout1.close()




def ID_write_head(xlspath):
    fout1 = open(xlspath, 'w')
    fout1.write("".join(['ID\tOrgan']) + '\n')
    fout1.close()

def ID_write(xlspath, ID):
    fout1 = open(xlspath, mode='a')
    fout1.write("".join(ID) + '\n')
    fout1.close()

def radiomics_apply_LIRADS(main_path, pat, str_out, data_site, registered):

    print(pat, ' ...\n')
    organ = 'liver'
    phases = ['pre', 'A', 'V', 'D']


    file_path = main_path + "/" + pat
    path_radiomics = file_path + '/radiomics.csv'


    result_list = []

    for phase in phases:
        imageName = file_path + '/img_' + phase + '.nii.gz'

        maskName = file_path + '/' + 'C_D.nii.gz'


        featureVector = radiomics_process(imageName, maskName, 1)
        result_list.append(featureVector)

    namelist, feature_list_all = xls_write(result_list, path_radiomics, pat)

    ID_write('Processed_ID.xls', [pat, '\t', organ])

    result = [namelist, feature_list_all]


    global finished_count
    finished_count = finished_count + 1

    print(pat, ' done!\n')
    print('Current finished: ', str_out, '\n')
    print('Totally finished: ', str(finished_count), '\n')
    return result





multi_processors = False

finished_count = 0

data_site = ''
registered = True


if __name__=='__main__':

    main_path = r''
    out_path = './/'


    if not os.path.exists(out_path):
        os.mkdir(out_path)

    ID_write_head('not_processed_ID.xls')
    ID_write_head('Processed_ID.xls')
    ID_write_head('Processed_ID_last.xls')

    flag=True

    print('Parent process %s.' % os.getpid())
    p = Pool(5)
    dirs = sorted(os.listdir(main_path))


    results = []

    liver_out = []

    for ID in range(0, len(dirs)):
        pat = dirs[ID]
        print(pat)


        str_out = str(len(dirs)) + ', ' + str(ID+1)

        results.append(p.apply_async(radiomics_apply_LIRADS, args=(main_path, pat, str_out, data_site, registered)))
        # result = radiomics_apply_LIRADS(main_path, pat, str_out, data_site, registered)



    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


    for result in results:
        liver_result= result.get()
        if liver_result != 0:
            liver_out.append(liver_result)

    if len(liver_out) > 0:
        xls_write_all(liver_out, out_path + '/liver.csv')
        xls_write_4phases(liver_out, out_path, 'liver')

    print('All Done!')


