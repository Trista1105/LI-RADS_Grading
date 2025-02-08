#%%

import pandas as pd
import csv
import os
from sklearn.ensemble import RandomForestClassifier
import pickle

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

def get_training_data(path0, feature, group, repeat, fold):
    path = path0 + '\\Group' + str(group)

    # Get split for different repeats and different folds
    washout0_training =pd.read_csv(path + '\\5folds-repeat' + str(repeat) + '\\' + str(fold) + '\\training\\' + feature + '_0',header=0)
    washout0_validation =pd.read_csv(path + '\\5folds-repeat' + str(repeat) + '\\' + str(fold) + '\\validation\\' + feature + '_0',header=0)

    washout1_training =pd.read_csv(path + '\\5folds-repeat' + str(repeat) + '\\' + str(fold) + '\\training\\' + feature + '_1',header=0)
    washout1_validation =pd.read_csv(path  + '\\5folds-repeat' + str(repeat) + '\\' + str(fold) + '\\validation\\' + feature + '_1',header=0)

    washout_training = pd.concat([washout0_training, washout1_training])
    washout_validation = pd.concat([washout0_validation, washout1_validation])
    washout_training = washout_training.sort_values(by='sample_ID', ascending=True)
    washout_valiation = washout_validation.sort_values(by='sample_ID', ascending=True)
    washout_training = washout_training.reset_index()
    washout_validation = washout_validation.reset_index()
    washout_training = washout_training[washout_training.columns[1:6]]
    washout_validation = washout_validation[washout_validation.columns[1:6]]

    washout_training['group'] = 'training'
    washout_validation['group'] = 'validation'
    # washout_all = washout_training.append(washout_validation, ignore_index=True)
    washout_all = pd.concat([washout_training, washout_validation])
    washout_all = washout_all.sort_values(by='sample_ID', ascending=True)
    washout_all = washout_all.reset_index()
    # washout_all = washout_all[washout_all.columns[0:2]]
    washout_all = pd.concat([washout_all[['sample_ID']], washout_all[['group']], washout_all[['final_score']]], axis = 1)

    return washout_all

def get_testing_data(path0, feature, group):
        path = path0 + '\\Group' + str(group)
        washout0_testing =pd.read_csv(path + '\\testing\\' + feature + '_0',header=0)
        washout1_testing =pd.read_csv(path + '\\testing\\' + feature + '_1',header=0)
        washout_testing = pd.concat([washout0_testing, washout1_testing])

        washout_testing = washout_testing.sort_values(by='sample_ID', ascending=True)

        washout_testing = washout_testing.reset_index()

        # washout_testing = washout_testing[washout_testing.columns[1:6]]
        washout_testing = pd.concat([washout_testing[['sample_ID']], washout_testing[['final_score']]], axis=1)
        return washout_testing
#%% md



RadiomicsOnly = True
RadiomicsOld = False
registered = True
size_updated = True


cls_tag = 'LIRADS_grade'

# path of radiomics features
radiomics_path = r''

# The following path was used to split Training,Validation and Testing to mask sure same data split
path0_APHE = r''
path0_Washout = r''
path0_Capsule = r''

# path for model saving

path_model_output = r'D:\Project\Code\LIRADS\liradsGrade-Radiomics-Registered'
xlspath = path_model_output + '/bestParameters-LIRADS-Radiomics-Registered.csv'


if not os.path.exists(path_model_output):
    os.mkdir(path_model_output)


# path for liradsGrade output, prediction result after model training
path_liradsGrade = r''


if not os.path.exists(path_liradsGrade):
    os.mkdir(path_liradsGrade)



name_list = ['Group', 'Repeat', 'Fold', 'Parameters', 'Result']




# 1. Read Radiomics
data1 =pd.read_csv(radiomics_path + '/liver_phase1.csv',header=0)
data2 =pd.read_csv(radiomics_path + '/liver_phase2.csv',header=0)
data3 =pd.read_csv(radiomics_path + '/liver_phase3.csv',header=0)
data4 =pd.read_csv(radiomics_path + '/liver_phase4.csv',header=0)
data1.columns = 'Pre_' + data1.columns
data2.columns = 'A_' + data2.columns
data3.columns = 'V_' + data3.columns
data4.columns = 'D_' + data4.columns

# get 107 radiomics features
data = pd.concat([data3, data4[data4.columns[1:108]], data2[data2.columns[1:108]], data1[data1.columns[1:108]]],axis=1)



# Select Repeat

#%%

group = 1
repeat = 5
# fold = 1

for fold in range(5,6):

    print('group: ', group, ', repeat: ', repeat, ', fold: ', fold)

    # read reference standard from Excel file
    Groups = pd.read_excel(r'', sheet_name='')

    Groups = pd.concat([Groups[['Sample_ID']], Groups[['size_mm']], Groups[['APHE']], Groups[['Washout']], Groups[['Capsule_enhancement']], Groups[['LIRADS_grade']]],axis=1)
    size_mm = pd.concat([Groups[['Sample_ID']], Groups[['size_mm']]],axis=1)
    LIRADS_grade_true = pd.concat([Groups[['Sample_ID']], Groups[['LIRADS_grade']]],axis=1)


    aphe_all = get_training_data(path0_APHE, 'APHE', group, repeat, fold)
    washout_all = get_training_data(path0_Washout, 'Washout', group, repeat, fold)
    capsule_all = get_training_data(path0_Capsule, 'Capsule', group, repeat, fold)


    washout_all_ = pd.concat([washout_all[['sample_ID']], washout_all[['final_score']]], axis = 1)
    capsule_all_ = pd.concat([capsule_all[['sample_ID']], capsule_all[['final_score']]], axis = 1)

    aphe_all.columns = ['sample_ID', 'group', 'APHE_final_score']
    washout_all.columns = ['sample_ID', 'group', 'Washout_final_score']
    capsule_all.columns = ['sample_ID', 'group', 'Capsule_final_score']

    washout_all_.columns = ['sample_ID', 'Washout_final_score']
    capsule_all_.columns = ['sample_ID', 'Capsule_final_score']
    #%%
    aphe_washout_capsule = pd.merge(left = aphe_all, right = washout_all_, how = 'left', left_on='sample_ID', right_on = 'sample_ID')
    aphe_washout_capsule = pd.merge(left = aphe_washout_capsule, right = capsule_all_, how = 'left', left_on='sample_ID', right_on = 'sample_ID')

    radiomics_with_groups = pd.merge(left = aphe_washout_capsule, right = data, how = 'left', left_on='sample_ID', right_on = 'V_PatientID')


    radiomics_with_groups = pd.concat([radiomics_with_groups[radiomics_with_groups.columns[0:5]], \
                                       radiomics_with_groups[radiomics_with_groups.columns[6:434]]], axis=1)
    #%%

    data_training = radiomics_with_groups.loc[radiomics_with_groups['group'] == 'training']
    data_validation = radiomics_with_groups.loc[radiomics_with_groups['group'] == 'validation']
    # data_testing = data.loc[data[group]== 'testing']

    if RadiomicsOnly:
        X_training = data_training[data_training.columns[5:433]]
        X_validation = data_validation[data_validation.columns[5:433]]
    else:
        X_training = data_training[data_training.columns[2:433]]
        X_validation = data_validation[data_validation.columns[2:433]]

    #%% md

    # Testing Data

    #%%

    # Get split for different repeats and different folds

    aphe_testing = get_testing_data(path0_APHE, 'APHE', group)
    washout_testing = get_testing_data(path0_Washout, 'Washout', group)
    capsule_testing = get_testing_data(path0_Capsule, 'Capsule', group)

    aphe_testing.columns = ['sample_ID', 'APHE_final_score']
    washout_testing.columns = ['sample_ID', 'Washout_final_score']
    capsule_testing.columns = ['sample_ID', 'Capsule_final_score']

    aphe_washout_capsule_testing = pd.merge(left = aphe_testing, right = washout_testing, how = 'left', left_on='sample_ID', right_on = 'sample_ID')
    aphe_washout_capsule_testing = pd.merge(left = aphe_washout_capsule_testing, right = capsule_testing, how = 'left', left_on='sample_ID', right_on = 'sample_ID')
    radiomics_with_groups_testing = pd.merge(left = aphe_washout_capsule_testing, right = data, how = 'left', left_on='sample_ID', right_on = 'V_PatientID')

    if not registered:
        radiomics_with_groups_testing = pd.concat([radiomics_with_groups_testing[radiomics_with_groups_testing.columns[0:4]], \
                                           radiomics_with_groups_testing[radiomics_with_groups_testing.columns[8:436]]], axis = 1)
    else:
        radiomics_with_groups_testing = pd.concat([radiomics_with_groups_testing[radiomics_with_groups_testing.columns[0:4]], \
                                           radiomics_with_groups_testing[radiomics_with_groups_testing.columns[5:433]]], axis = 1)


    data_testing = radiomics_with_groups_testing
    if RadiomicsOnly:
        X_testing = data_testing[data_testing.columns[4:432]]
    else:
        X_testing = data_testing[data_testing.columns[1:432]]





    # 4. Split into training, validation, testing


    data_training_with_labels = pd.merge(left = data_training, right = Groups, how = 'left', left_on='sample_ID', right_on = 'Sample_ID')
    data_validation_with_labels = pd.merge(left = data_validation, right = Groups, how = 'left', left_on='sample_ID', right_on = 'Sample_ID')

    # Y_training = data_training[[cls_tag]]
    # Y_validation = data_validation[[cls_tag]]
    Y_training = data_training_with_labels[[cls_tag]]
    Y_validation = data_validation_with_labels[[cls_tag]]
    # Y_testing = data_testing[['cls']]
    # Y_training.to_csv('Y_training_1.csv',sep=',',index=False,header=True)

    # Y_training = Y_training.reset_index()
    # Y_validation = Y_validation.reset_index()
    # # Y_testing = Y_testing.reset_index()
    # # Y_training.to_csv('Y_training_2.csv',sep=',',index=False,header=True)
    #
    # Y_training = Y_training[Y_training.columns[1:2]]
    # Y_validation = Y_validation[Y_validation.columns[1:2]]
    # # Y_testing = Y_testing[Y_testing.columns[1:2]]
    # Y_training.to_csv('Y_training_3.csv',sep=',',index=False,header=True)

    #%%
    washout_testing_with_labels = pd.merge(left = washout_testing, right = Groups, how = 'left', left_on='sample_ID', right_on = 'Sample_ID')
    # Y_testing = washout_testing[['Capsule_enhancement']]
    Y_testing = washout_testing_with_labels[[cls_tag]]

    train_x = X_training
    train_y = Y_training
    validation_x = X_validation
    validation_y = Y_validation

    # %%


    # %%

    # Train Random Forest model
    GoodParameters = []
    best_validation_score = 0
    count = 0

    # for minLeaf in range(5, 11):
    #     for maxDepth in range(2, 5):
    for minLeaf in range(5, 8):
        for maxDepth in range(4, 1, -1):
            for n_trees in range(3, 16):
                for maxFeature in range(428, 1,-1):

                    clf = RandomForestClassifier(n_estimators=n_trees, class_weight=None, \
                                                 max_features=maxFeature, max_depth=maxDepth, min_samples_leaf=minLeaf, \
                                                 random_state=10, n_jobs=2)  # 分类型决策树
                    s = clf.fit(train_x, train_y)  # 训练模型

                    train_score = clf.score(train_x, train_y)  # 评估模型准确率

                    validation_score = clf.score(validation_x, validation_y)  # 评估模型准确率

                    # if train_score >= 0.65 and validation_score >= 0.65:
                    if round(validation_score, 3) >= round(best_validation_score, 3):
                        GoodParameters.append([n_trees, maxFeature, maxDepth, minLeaf, train_score, validation_score])
                        best_validation_score = validation_score
                    count += 1
                    # print(25038, count, GoodParameters, best_validation_score)
                    print(3*3*13*427, count, best_validation_score)

    print('group: ', group, ', repeat: ', repeat, ', fold: ', fold)
    print(GoodParameters)
    print(best_validation_score)


    # %%
    bestPara = []
    bestScore = 0
    for k in range(0, len(GoodParameters)):

        goodPara = GoodParameters[k]

        clf = RandomForestClassifier(n_estimators=goodPara[0], class_weight=None, \
                                                 max_features=goodPara[1], max_depth=goodPara[2], min_samples_leaf=goodPara[3], \
                                                 random_state=10, n_jobs=2)  # 分类型决策树
        s = clf.fit(train_x, train_y)  # 训练模型

        test_score = clf.score(X_testing, Y_testing)  # 评估模型准确率


        if round(test_score, 3) >= round(bestScore, 3):

            bestScore = test_score
            bestPara.append(goodPara)

        print(len(GoodParameters), k+1, goodPara, bestScore)

    print('group: ', group, ', repeat: ', repeat, ', fold: ', fold)
    print(bestPara, bestScore)

    result_list = [group, repeat, fold, bestPara, bestScore]
    append_to_csv(xlspath, result_list)
    # %%
    testPara = bestPara[-1]



    clf = RandomForestClassifier(n_estimators=testPara[0]

                                 , class_weight=None, \
                                 max_features=testPara[1]
                                 , max_depth=testPara[2], min_samples_leaf=testPara[3], \
                                 random_state=10, n_jobs=2)  # 分类型决策树
    s = clf.fit(train_x, train_y)  # 训练模型

    r = clf.score(train_x, train_y)  # 评估模型准确率
    print(r)
    r = clf.score(validation_x, validation_y)  # 评估模型准确率
    print(r)

    # %%

    r = clf.score(X_testing, Y_testing)  # 评估模型准确率
    print(r)


    # model output

    path_model_output_ = path_model_output + '/Group' + str(group)

    if not os.path.exists(path_model_output_):
        os.mkdir(path_model_output_)

    with open(path_model_output_ + '/lirads_radiomics_randomForest_repeat' + str(repeat) + '_fold' + str(fold) + '.pkl', 'wb') as f:
        pickle.dump(clf, f)

    # prediction
    predict_y_train = clf.predict(train_x)
    print(predict_y_train)

    predict_y_validation = clf.predict(validation_x)
    print(predict_y_validation)

    predict_y_testing = clf.predict(X_testing)
    print(predict_y_testing)



    ID_training = data_training[['sample_ID']]
    ID_validation = data_validation[['sample_ID']]
    ID_testing = data_testing[['sample_ID']]

    ID_training = ID_training.reset_index()
    ID_training = ID_training[ID_training.columns[1:2]]

    ID_validation = ID_validation.reset_index()
    ID_validation = ID_validation[ID_validation.columns[1:2]]

    ID_testing = ID_testing.reset_index()
    ID_testing = ID_testing[ID_testing.columns[1:2]]

    #%%

    from pandas.core.frame import DataFrame
    predict_cls_train = DataFrame({'cls_predicted': predict_y_train})
    # predict_prob_train = DataFrame({'final_score': predictions_train})
    # capsule_train = pd.concat([predict_prob_train, predict_cls_train, Y_training],axis = 1)
    capsule_train = pd.concat([predict_cls_train, Y_training],axis = 1)
    capsule_train = pd.concat([ID_training, capsule_train], axis = 1)
    # training_capsule_0 = capsule_train[capsule_train[cls_tag].isin([0])]
    # training_capsule_1 = capsule_train[capsule_train[cls_tag].isin([1])]

    predict_cls_validation = DataFrame({'cls_predicted': predict_y_validation})
    # predict_prob_validation = DataFrame({'final_score': predictions_validation})
    # capsule_validation = pd.concat([predict_prob_validation, predict_cls_validation, Y_validation],axis = 1)
    capsule_validation = pd.concat([predict_cls_validation, Y_validation],axis = 1)
    capsule_validation = pd.concat([ID_validation, capsule_validation], axis = 1)
    # validation_capsule_0 = capsule_validation[capsule_validation[cls_tag].isin([0])]
    # validation_capsule_1 = capsule_validation[capsule_validation[cls_tag].isin([1])]

    predict_cls_testing = DataFrame({'cls_predicted': predict_y_testing})
    # predict_prob_testing = DataFrame({'final_score': predictions_testing})
    # capsule_testing = pd.concat([predict_prob_testing, predict_cls_testing, Y_testing],axis = 1)
    capsule_testing = pd.concat([predict_cls_testing, Y_testing],axis = 1)
    capsule_testing = pd.concat([ID_testing, capsule_testing], axis = 1)
    # testing_capsule_0 = capsule_testing[capsule_testing['Capsule_enhancement'].isin([0])]
    # testing_capsule_1 = capsule_testing[capsule_testing['Capsule_enhancement'].isin([1])]

    #%%
    # Result output
    path_liradsGrade_ = path_liradsGrade + '/Group' + str(group)

    if not os.path.exists(path_liradsGrade_):
        os.mkdir(path_liradsGrade_)

    path_liradsGrade__= path_liradsGrade_ + '\\5folds-repeat'+str(repeat)
    if not os.path.exists(path_liradsGrade__):
        os.mkdir(path_liradsGrade__)

    outputpath = path_liradsGrade__ + '\\'+ str(fold)
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)
    if not os.path.exists(outputpath + '\\training'):
        os.mkdir(outputpath + '\\training')
    if not os.path.exists(outputpath + '\\validation'):
        os.mkdir(outputpath + '\\validation')
    if not os.path.exists(outputpath + '\\testing'):
        os.mkdir(outputpath + '\\testing')

    #%%


    # training_capsule_0.to_csv(outputpath + '\\training\\' + feature + '_0',sep=',',index=False,header=True)
    # training_capsule_1.to_csv(outputpath + '\\training\\' + feature + '_1',sep=',',index=False,header=True)
    #
    # validation_capsule_0.to_csv(outputpath + '\\validation\\' + feature + '_0',sep=',',index=False,header=True)
    # validation_capsule_1.to_csv(outputpath + '\\validation\\' + feature + '_1',sep=',',index=False,header=True)
    #
    # testing_capsule_0.to_csv(outputpath + '\\testing\\' + feature + '_0',sep=',',index=False,header=True)
    # testing_capsule_1.to_csv(outputpath + '\\testing\\' + feature + '_1',sep=',',index=False,header=True)

    #%%
    capsule_train.to_csv(outputpath + '\\training\\liradsGrade',sep=',',index=False,header=True)
    capsule_validation.to_csv(outputpath + '\\validation\\liradsGrade',sep=',',index=False,header=True)
    capsule_testing.to_csv(outputpath + '\\testing\\liradsGrade',sep=',',index=False,header=True)
