#%%

import pandas as pd
import csv
import os
from sklearn.ensemble import RandomForestClassifier

def create_csv_with_header(xlspath, name_list):
    fout1 = open(xlspath, 'w', newline='' '')
    csv_writer = csv.writer(fout1)

    csv_writer.writerow(name_list)
    fout1.close()

def append_to_csv(xlspath, result_list):
    fout1 = open(xlspath, 'a', newline='' '')
    csv_writer = csv.writer(fout1)

    csv_writer.writerow(result_list)
    fout1.close()

def get_testing_data(path, feature):

    washout0_testing =pd.read_csv(path + '/' + feature + '_0',header=0)
    washout1_testing =pd.read_csv(path + '/' + feature + '_1',header=0)
    washout_testing = pd.concat([washout0_testing, washout1_testing])

    washout_testing = washout_testing.sort_values(by='sample_ID', ascending=True)

    washout_testing = washout_testing.reset_index()

    # washout_testing = washout_testing[washout_testing.columns[1:6]]
    washout_testing = pd.concat([washout_testing[['sample_ID']], washout_testing[['final_score']]], axis=1)
    return washout_testing

data_site = ''

registered = True
size_updated = True

RadiomicsOnly = True
cls_tag = 'LIRADS_grade'

# 1. Read Radiomics

radiomics_path = r''


data1 = pd.read_csv(radiomics_path + '/liver_phase1.csv', header=0)
data2 = pd.read_csv(radiomics_path + '/liver_phase2.csv', header=0)
data3 = pd.read_csv(radiomics_path + '/liver_phase3.csv', header=0)
data4 = pd.read_csv(radiomics_path + '/liver_phase4.csv', header=0)

data1.columns = 'Pre_' + data1.columns
data2.columns = 'A_' + data2.columns
data3.columns = 'V_' + data3.columns
data4.columns = 'D_' + data4.columns
# data = pd.concat([data3, data4[data4.columns[4:111]]],axis=1)
data = pd.concat([data3, data4[data4.columns[1:108]], data2[data2.columns[1:108]], data1[data1.columns[1:108]]],axis=1)
data.rename(columns={'V_PatientID':'sample_ID'}, inplace=True)


# Radom Forest Model path

path3 = r''



# LI-RADS Grade outpath
path1_ = r''


if not os.path.exists(path1_):
    os.mkdir(path1_)


# read LI-RADS Grade True Label accordingly

Groups = pd.read_excel(r'.xlsx', sheet_name='')

Groups = Groups.sort_values(by='Sample_ID', ascending=True)
Groups = Groups.reset_index()
Groups = pd.concat([Groups[['Sample_ID']], Groups[['size_mm']], Groups[['APHE']], Groups[['Washout']], Groups[['Capsule_enhancement']], Groups[['LIRADS_grade']]],axis=1)
size_mm = pd.concat([Groups[['Sample_ID']], Groups[['size_mm']]],axis=1)
LIRADS_grade_true = pd.concat([Groups[['Sample_ID']], Groups[['LIRADS_grade']]],axis=1)

#%%

data_testing = pd.merge(left=Groups, right=data, how='left', left_on='Sample_ID', right_on='sample_ID')
X_testing = data_testing[data_testing.columns[7:435]]




washout_testing_with_labels = pd.merge(left=data, right=Groups, how='right', left_on='sample_ID',
                                           right_on='Sample_ID')
Y_testing = washout_testing_with_labels[[cls_tag]]



for group in range(1,2):
    for repeat in range(1,6):
        for fold in range(1,6):

            from sklearn.ensemble import RandomForestClassifier
            import pickle
            pkl_file = open(path3 + '/Group' + str(group) + '/lirads_radiomics_randomForest_repeat' + str(repeat) + '_fold' + str(fold) + '.pkl', 'rb')

            clf = pickle.load(pkl_file)


            r = clf.score(X_testing, Y_testing)  # 评估模型准确率
            print(r)


            predict_y_testing = clf.predict(X_testing)#直接给出预测结果，每个点在所有label的概率和为1，内部还是调用predict——proba()
            # print(predict_y_testing)
            prob_predict_y_testing = clf.predict_proba(X_testing)#给出带有概率值的结果，每个点所有label的概率和为1
            predictions_testing = prob_predict_y_testing[:, 1]


            ID_testing = data_testing[['sample_ID']]


            ID_testing = ID_testing.reset_index()
            ID_testing = ID_testing[ID_testing.columns[1:2]]


            #%%

            from pandas.core.frame import DataFrame

            predict_cls_testing = DataFrame({'cls_predicted': predict_y_testing})
            capsule_testing = pd.concat([predict_cls_testing, Y_testing], axis=1)
            capsule_testing = pd.concat([ID_testing, capsule_testing], axis=1)


            # Result output


            path1 = path1_ + '\\Group' + str(group)
            if not os.path.exists(path1):
                os.mkdir(path1)
            path2 = path1 + '\\5folds-repeat' + str(repeat)
            if not os.path.exists(path2):
                os.mkdir(path2)

            outputpath = path2 + '\\' + str(fold)
            if not os.path.exists(outputpath):
                os.mkdir(outputpath)


            capsule_testing.to_csv(outputpath + '\\liradsGrade', sep=',', index=False, header=True)



########################################### Ensemble Model ##############################


for group in range(1,2):
    for repeat in range(1,6):

        subpath = path1_ + '\\Group' + str(group) + '\\5folds-repeat' + str(repeat)

        Capsules = []

        for fold in range(1,6):

            result_path = subpath + '\\' + str(fold)

            path_capsule_1 = result_path + '\\liradsGrade'


            capsule = pd.read_csv(path_capsule_1)


            Capsules.append(capsule)


        Capsule = pd.concat(Capsules).groupby('sample_ID')['cls_predicted'].sum().reset_index()
        Capsule['cls_predicted']=Capsule['cls_predicted']/5

        Capsule_ = Capsule.round(0)
        Capsule_['cls_predicted'] = Capsule_['cls_predicted'].astype(int)


        Capsule_.to_csv(subpath  + '\\liradsGrade',sep=',',index=False,header=True)

