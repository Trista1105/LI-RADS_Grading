# group = 4
# repeat = 5
# fold = 5


# Early Stop
# Cross Validation


# %matplotlib inline
# import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout
# from keras.applications import VGG16
from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

import sklearn
# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix

from keras import backend as K

import keras
from keras.models import load_model
from keras.applications import vgg16
from keras import Input
from keras.layers.merge import concatenate

from keras.layers import Dropout

########################################################## AlexNet Model ########################################
from tensorflow.keras.layers import Conv2D, Lambda, MaxPool2D, Flatten, Dense, Dropout, Activation, \
    ZeroPadding2D, Input

#####################################################################
# model parameters
from keras import optimizers

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale

K.clear_session()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

tf.config.experimental_run_functions_eagerly(True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

import matplotlib.pyplot as plt

import csv
import os
from pandas.core.frame import DataFrame


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




def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]


def print_confusion_matrix(cls_test, cls_pred, class_names):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    print("Confusion matrix:")

    # Print the confusion matrix as text.
    print(cm)

    # Print the class-names for easy reference.
    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))


def plot_training_history(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get it for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')

    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()


def print_layer_trainable_vgg(vgg_model):
    for layer in vgg_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

def print_layer_trainable_VGG(VGG_model):
    for layer in VGG_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

def print_layer_trainable_LIRADS(LIRADS_model):
    for layer in LIRADS_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))



def data_subgroup(Array, Index, ifAug):
    Array_sub = []
    for i in range(0, len(Index)):
        index = Index[i]
        if ifAug:
            for j in range(0, 7):
                Array_index = 7 * index + j
                Array_sub.append(Array[Array_index])
        else:
            Array_index = 7 * index
            Array_sub.append(Array[Array_index])

    return Array_sub


def data_copy(Array, k):
    Array_Copy = []
    for i in range(0, Array.shape[0]):
        array = Array[i]
        for j in range(0, k):
            Array_Copy.append(array)

    return Array_Copy


# class_weight = {0:3,1:3,2:1}
class_weight = {0: 1, 1: 1, 2: 1}


################################ Model #####################################
num_classes = 3
fourFeatures = False
model = 'VGG16'
home1 = True

if not home1:
    model_subpath = '/VGG16/log'
else:
    model_subpath = ''

if model == 'VGG16':
    batch_size = 128  # VGG16 Paper:32,  AlexNet Paper:8
    learning_rate = 1e-4
if model == 'AlexNet':
    batch_size = 128
    learning_rate = 1e-5 # 1e-4

epochs = 250 #250  # VGG16 Paper: 400+400, AlexNet Paper: 200, stop at 53
patience = 15


ifAug = True


path = '/' # root path to save model
modeloutpath = path + '/Result/VGG16'

if not os.path.exists(modeloutpath):
    os.mkdir(modeloutpath)

xlspath = path + '/Result/' + model + '/LIRADS-VGG.csv'


name_list = ['Group', 'Repeat', 'Fold', 'Training Loss', 'Validation Loss', 'Testing Loss', 'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy', 'Best Epoch']
create_csv_with_header(xlspath, name_list)




# Auto judge if training needs to stop



# 1. Read LIRADS Grade Label

Labels = pd.read_excel(path + '//.xlsx', sheet_name='')
Labels = pd.concat([Labels[['Sample_ID']], Labels[['size_mm']], Labels[['APHE']], Labels[['Washout']],
                    Labels[['Capsule_enhancement']], Labels[['LIRADS_grade']]], axis=1)

size_mm = pd.concat([Labels[['Sample_ID']], Labels[['size_mm']]], axis=1)
LIRADS_grade_true = pd.concat([Labels[['Sample_ID']], Labels[['LIRADS_grade']]], axis=1)

# 2. Read IDs and ID Index for all the data

IDs = pd.read_csv(path + '/Data/IDs.csv', header=0)




#################################### VGG16 Model #####################################

model_vgg16 = keras.models.load_model('./vgg16/vgg16.h5')

vgg_final_layer = model_vgg16.get_layer('flatten')
vgg_model = Model(inputs=model_vgg16.input,
                  outputs=vgg_final_layer.output)

# Diameter Branch
diameter_input = Input(shape=(1,), name='diameter_input')
diameter_model = Dense(32, name='diameter_FC')(diameter_input)
diameter_model = Model(inputs=diameter_input, outputs=diameter_model)

# LIRADS 4 features branch
fourfeatures_input = Input(shape=(4,), name='4features_input')
fourfeatures_model = Dense(32, name='4features_FC')(fourfeatures_input)
fourfeatures_model = Model(inputs=fourfeatures_input, outputs=fourfeatures_model)

vgg_diameter = concatenate([diameter_model.output, vgg_model.output])
vgg_4features = concatenate([fourfeatures_model.output, vgg_model.output])

FC = Dense(1024, activation='relu')(vgg_diameter)
FC = Dense(512, activation='relu')(FC)
FC = Dropout(0.5)(FC)
FC = Dense(num_classes, activation='softmax')(FC)

FC4features = Dense(1024, activation='relu')(vgg_4features)
FC4features = Dense(512, activation='relu')(FC4features)
FC4features = Dropout(0.5)(FC4features)
FC4features = Dense(num_classes, activation='softmax')(FC4features)

FC1 = Dense(1024, activation='relu')(vgg_model.output)
FC1 = Dense(512, activation='relu')(FC1)
FC1 = Dropout(0.5)(FC1)
FC1 = Dense(num_classes, activation='softmax')(FC1)



# print_layer_trainable_vgg(vgg_model)
# print_layer_trainable_VGG(VGG_model)
# print_layer_trainable_LIRADS(LIRADS_model)

for layer in vgg_model.layers:
    layer.trainable = False

for layer in vgg_model.layers:
    # Boolean whether this layer is trainable.
    # trainable = ('block5' in layer.name or 'block4' in layer.name or 'flatten' in layer.name)
    trainable = ('block' in layer.name or 'flatten' in layer.name)

    # Set the layer's bool.
    layer.trainable = trainable

# print_layer_trainable_vgg(vgg_model)
# print_layer_trainable_VGG()
# print_layer_trainable_LIRADS()

#################################### End VGG16 Model #####################################

#################################### Model Optimization ##################################
# sgd = optimizers.gradient_descent_v2.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
# sgd = optimizers.gradient_descent_v2.SGD(lr=4e-5, momentum=0, nesterov=False)

# learning rate
# VGG16 Paper: 4e-5~4e-4, decay=0.9, cycle length=3, factor*cycle length, factor = 2
# AlexNet Paper: 1e-5~4e-2, clinical learning rate, triangular waveform

# sgd = optimizers.SGD(lr=1e-6, decay=0, momentum=0.2, nesterov=True)  # 1e-5, 1e-6 # AttributeError: module 'keras.optimizers' has no attribute 'SGD'
sgd = tf.keras.optimizers.SGD(lr=1e-6, decay=0, momentum=0.2, nesterov=True)  # 1e-5, 1e-6
# LearningRate = LearningRate * 1/(1 + decay * epoch)
# VGG16 Paper: SGD with momentum, SGD with restarts

adam = Adam(lr=learning_rate)  # 1e-5, 1e-6

loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']

###########################################################################


# VGG16, with Diameter or without Diameter
if __name__ == '__main__':
    withDiameter = True

    for group in range(1, 2):
        for repeat in range(1, 2):
            for fold in range(1, 6):

                print('group: ', group, ', repeat: ', repeat, ', fold: ', fold)
                if not home1:
                    subpath = path + '/Data/Groups/Group' + str(group) + '/5folds-repeat' + str(repeat) + '/' + str(fold) #可以是其他方法的结果，只是用于获取真实的LI-RADS Grade Label
                else:
                    subpath = path + '/LI-RADS-label/Group' + str(group) + '/5folds-repeat' + str(repeat) + '/' + str(fold)  # 可以是其他方法的结果，只是用于获取真实的LI-RADS Grade Label; home1目录下使用的是Radiomics的LI-RADS Grade结果文件

                IDs_training = pd.read_csv(subpath + '/training/liradsGrade', header=0)
                IDs_validation = pd.read_csv(subpath + '/validation/liradsGrade', header=0)
                IDs_testing = pd.read_csv(subpath + '/testing/liradsGrade', header=0)
                IDs_training = pd.concat([IDs_training[['sample_ID']], IDs_training[['LIRADS_grade']]], axis=1)
                IDs_validation = pd.concat([IDs_validation[['sample_ID']], IDs_validation[['LIRADS_grade']]], axis=1)
                IDs_testing = pd.concat([IDs_testing[['sample_ID']], IDs_testing[['LIRADS_grade']]], axis=1)

                # 4. Add ID Index to training, validation and testing
                IDs_training_with_index = pd.merge(left=IDs_training, right=IDs, how='left', left_on='sample_ID',
                                                   right_on='sample_ID')
                IDs_validation_with_index = pd.merge(left=IDs_validation, right=IDs, how='left', left_on='sample_ID',
                                                     right_on='sample_ID')
                IDs_testing_with_index = pd.merge(left=IDs_testing, right=IDs, how='left', left_on='sample_ID',
                                                  right_on='sample_ID')

                # 5. Read image pixel value data
                # Pre_Array = np.load(path + '/Data/Pre_Array_Aug.npy')
                # A_Array = np.load(path + '/Data/A_Array_Aug.npy')
                # V_Array = np.load(path + '/Data/V_Array_Aug.npy')
                # D_Array = np.load(path + '/Data/D_Array_Aug.npy')
                if not home1:
                    Pre_Array = np.load(path + '/Data/Pre_Array_Patches_Scaled_Aug.npy')
                    A_Array = np.load(path + '/Data/A_Array_Patches_Scaled_Aug.npy')
                    V_Array = np.load(path + '/Data/V_Array_Patches_Scaled_Aug.npy')
                    D_Array = np.load(path + '/Data/D_Array_Patches_Scaled_Aug.npy')
                else:
                    Pre_Array = np.load(path + '/npy/ZhongShan/Pre_Array_Patches_Scaled_Aug.npy')
                    A_Array = np.load(path + '/npy/ZhongShan/A_Array_Patches_Scaled_Aug.npy')
                    V_Array = np.load(path + '/npy/ZhongShan/V_Array_Patches_Scaled_Aug.npy')
                    D_Array = np.load(path + '/npy/ZhongShan/D_Array_Patches_Scaled_Aug.npy')

                # Array_Pre_A_V_D = np.concatenate([V_Array, D_Array, A_Array],axis=1)
                Array_Pre_A_V_D = np.zeros((A_Array.shape[0], 224, 224, 3))
                Array_Pre_A_V_D[:, :, :, 0] = V_Array
                Array_Pre_A_V_D[:, :, :, 1] = D_Array
                Array_Pre_A_V_D[:, :, :, 2] = A_Array

                # 6. Split image pixel value data into training, validation and testing
                training_index = IDs_training_with_index['Index']
                training_index_ = training_index.values.tolist()
                validation_index = IDs_validation_with_index['Index']
                validation_index_ = validation_index.values.tolist()
                testing_index = IDs_testing_with_index['Index']
                testing_index_ = testing_index.values.tolist()

                Array_training = np.array(data_subgroup(Array_Pre_A_V_D, training_index_, ifAug))
                Array_validation = np.array(data_subgroup(Array_Pre_A_V_D, validation_index_, False))
                Array_testing = np.array(data_subgroup(Array_Pre_A_V_D, testing_index_, False))

                # Normalization
                # scaler = MinMaxScaler()
                # Array_training_scaled = scaler.fit_transform(Array_training)
                # Array_validation_scaled = scaler.fit_transform(Array_validation)
                # Array_testing_scaled = scaler.fit_transform(Array_testing)
                #
                # Array_training = Array_training_scaled.reshape((Array_training.shape[0], 224, 224, 3))
                # Array_validation = Array_validation_scaled.reshape((Array_validation.shape[0], 224, 224, 3))
                # Array_testing = Array_testing_scaled.reshape((Array_testing.shape[0], 224, 224, 3))

                # 7. LIRADS Grade: training, validation and testing
                LIRADS_grade_cls_training = np.array(IDs_training['LIRADS_grade'].values.tolist())
                if ifAug:
                    LIRADS_grade_cls_training = np.array(data_copy(LIRADS_grade_cls_training, 7))

                LIRADS_grade_cls_validation = np.array(IDs_validation['LIRADS_grade'].values.tolist())
                LIRADS_grade_cls_testing = np.array(IDs_testing['LIRADS_grade'].values.tolist())

                LIRADS_grade_training = to_categorical(LIRADS_grade_cls_training - 3)
                LIRADS_grade_validation = to_categorical(LIRADS_grade_cls_validation - 3)
                LIRADS_grade_testing = to_categorical(LIRADS_grade_cls_testing - 3)

                # 8. size_mm: training, validation and testing
                IDs_training_size_mm = pd.merge(left=IDs_training, right=size_mm, how='left', left_on='sample_ID',
                                                right_on='Sample_ID')
                IDs_training_size_mm_ = np.array(IDs_training_size_mm['size_mm'].values.tolist())
                if ifAug:
                    IDs_training_size_mm__ = np.array(data_copy(IDs_training_size_mm_, 7))
                IDs_validation_size_mm = pd.merge(left=IDs_validation, right=size_mm, how='left', left_on='sample_ID',
                                                  right_on='Sample_ID')
                IDs_testing_size_mm = pd.merge(left=IDs_testing, right=size_mm, how='left', left_on='sample_ID',
                                               right_on='Sample_ID')
                IDs_validation_size_mm_ = np.array(IDs_validation_size_mm['size_mm'].values.tolist())
                IDs_testing_size_mm_ = np.array(IDs_testing_size_mm['size_mm'].values.tolist())

                # 9. size_mm: normalization
                if ifAug:
                    size_mm_training_scaled = scale(IDs_training_size_mm__)

                else:
                    size_mm_training_scaled = scale(IDs_training_size_mm_)

                mean_training = np.mean(IDs_training_size_mm_)
                std_training = np.std(IDs_training_size_mm_)

                size_mm_validation_scaled = (IDs_validation_size_mm_ - mean_training) / std_training
                size_mm_testing_scaled = (IDs_testing_size_mm_ - mean_training) / std_training

                # 10. LIRADS 4 features together

                # For Cross Validation
                # X = np.concatenate([Array_training, Array_validation], axis=0)
                # Y = np.concatenate([LIRADS_grade_training, LIRADS_grade_validation], axis=0)
                # Y1 = [np.argmax(one_hot) for one_hot in Y]
                # Y1 = np.array(Y1)


                ############################################# Model ############################################################################################
                outpath = modeloutpath + '/Group' + str(group) + '/5folds-repeat' + str(repeat) + '/' + str(fold)

                checkpoint = keras.callbacks.ModelCheckpoint(
                    filepath=outpath + model_subpath + '/checkpoint/best_weights_epoch-{epoch:02d}-acc-{categorical_accuracy}-val_acc-{val_categorical_accuracy}.hdf5',
                    monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                    save_weights_only=False, mode='auto', period=1)
                tensorboard = keras.callbacks.TensorBoard(
                    log_dir=outpath + model_subpath + '/tensorboard',
                    write_graph=True,
                    write_images=True
                    #         histogram_freq=1,
                    #         embeddings_freq=1,
                    #         embeddings_data=x_train[:1000].astype('float32')
                )

                callbacks = [checkpoint, tensorboard]

                if not withDiameter:
                    VGG_model = Model(inputs=vgg_model.input, outputs=FC1)
                    LIRADS_Model = VGG_model
                else:
                    LIRADS_model = Model(inputs=[diameter_model.input, vgg_model.input], outputs=FC)
                    LIRADS_Model = LIRADS_model
                if fourFeatures:
                    LIRADS_model = Model(inputs=[fourfeatures_model.input, vgg_model.input], outputs=FC4features)
                    LIRADS_Model = LIRADS_model

                # LIRADS_Model.summary(line_length=150, positions=[0.30, 0.60, 0.7, 1.])


                
                LIRADS_Model.compile(optimizer=adam, loss=loss, metrics=metrics)

                ############################################# Model Training ############################################################################################

                bestValAcc = 0
                waite_epoch = 0
                waite_epoch_high_accuracy = 0
                last_10_val_loss = []
                best_epoch = 0



                for i in range(0, epochs):
                    print('epoch: ', str(i+1))
                    
                    if not withDiameter:

                        history = LIRADS_Model.fit(Array_training, LIRADS_grade_training, \
                                                   validation_data = (Array_validation, LIRADS_grade_validation), \
                                                   batch_size=batch_size, class_weight=class_weight,epochs=1, shuffle=True, verbose=1, callbacks=callbacks)
                    else:
                                               
                        history = LIRADS_Model.fit([size_mm_training_scaled, Array_training], LIRADS_grade_training, \
                                                   validation_data=([size_mm_validation_scaled, Array_validation],
                                                                    LIRADS_grade_validation), \
                                                   batch_size=batch_size, class_weight=class_weight, epochs=1,
                                                   shuffle=True, verbose=1, callbacks=callbacks)

                    train_acc = history.history['categorical_accuracy']
                    # acc = history.history['binary_accuracy']
                    train_loss = history.history['loss']

                    # Get it for the validation-set (we only use the test-set).
                    val_acc = history.history['val_categorical_accuracy']
                    # val_acc = history.history['val_binary_accuracy']
                    val_loss = history.history['val_loss']

                    if val_acc[0] >= 0.5 and val_acc[0] > bestValAcc:
                        bestValAcc = val_acc[0]
                        best_epoch = i + 1

                        LIRADS_Model.save_weights(outpath + model_subpath + '/model_weights_epochs-' + str(i+1) + \
                                                  '_train-loss_' + str(train_loss) + '_train-acc_' + str(train_acc) + '_val-loss_' + str(val_loss) + '_val-acc_' + str(val_acc) + '.h5')

                        LIRADS_Model.save(outpath + model_subpath + '/model_epochs-' + str(i+1) + '_train-loss_' + str(train_loss) + '_train-acc_' + str(train_acc) + '_val-loss_' + str(val_loss) + '_val-acc_' + str(val_acc) + '.h5')

                        LIRADS_Model.save_weights(outpath + model_subpath + '/vgg16_bestValAcc_weights.h5')

                        LIRADS_Model.save(outpath + model_subpath + '/vgg16_bestValAcc_model.h5')
                        LIRADS_Model.save(outpath + model_subpath + '/vgg16_bestValAcc_model.h')


                        waite_epoch = 0
                        waite_epoch_high_accuracy = 0

                    else:
                        if len(last_10_val_loss) > 0 and val_loss[0] - last_10_val_loss[-1] > 0.005:
                            waite_epoch += 1
                        elif train_acc[0] > 0.99 or val_acc[0] == 1:
                            waite_epoch_high_accuracy += 1


                    print('waite epochs: ', waite_epoch)
                    print('waite epochs for high accuracy: ', waite_epoch_high_accuracy)

                    if len(last_10_val_loss) < 10:
                        last_10_val_loss.append(val_loss[0])
                    else:
                        del (last_10_val_loss[0])
                        last_10_val_loss.append(val_loss[0])



                    if (waite_epoch >= patience) or (i + 1 > 20 and (val_loss[0]-train_loss[0]) > 0.5) or (train_acc[0] > 0.95 and val_acc[0] < 0.7) or waite_epoch_high_accuracy == 5:
                        i = epochs
                        break


                # LIRADS_Model.save(outpath + model_subpath + '/vgg16_final_epoch.h')
                LIRADS_Model.save(outpath + model_subpath + '/vgg16_final_epoch')
                LIRADS_Model.save(outpath + model_subpath + '/vgg16_final_epoch.h5')

                ############################################# End Model Training ############################################################################################

                ############################################# Model Prediction ############################################################################################
                # Load best model and do prediction
                if os.path.exists(outpath + model_subpath + '/vgg16_bestValAcc_model.h'):
                    LIRADS_Model = load_model(outpath + model_subpath + '/vgg16_bestValAcc_model.h')

                    if not withDiameter:
                        scores_training = LIRADS_Model.evaluate(Array_training, LIRADS_grade_training, verbose=1)
                        scores_validation = LIRADS_Model.evaluate(Array_validation, LIRADS_grade_validation, verbose=1)
                        scores_testing = LIRADS_Model.evaluate(Array_testing, LIRADS_grade_testing, verbose=1)
                    else:
                    
                        scores_training = LIRADS_Model.evaluate([size_mm_training_scaled, Array_training],
                                                                LIRADS_grade_training, verbose=1)
                        scores_validation = LIRADS_Model.evaluate([size_mm_validation_scaled, Array_validation],
                                                                  LIRADS_grade_validation, verbose=1)
                        scores_testing = LIRADS_Model.evaluate([size_mm_testing_scaled, Array_testing],
                                                               LIRADS_grade_testing, verbose=1)
                                                           

                    print(scores_training, scores_validation, scores_testing)

                    print('group: ', group, ', repeat: ', repeat, ', fold: ', fold)

                    result_list = [group, repeat, fold, scores_training[0], scores_validation[0], scores_testing[0], scores_training[1], scores_validation[1], scores_testing[1], best_epoch]
                    append_to_csv(xlspath, result_list)

                ################################################################################
                    # Prediction
                    if not withDiameter:
                        y_pred_training = LIRADS_Model.predict(Array_training)
                        y_pred_validation = LIRADS_Model.predict(Array_validation)
                        y_pred_testing = LIRADS_Model.predict(Array_testing)
                    else:
                        y_pred_training = LIRADS_Model.predict([size_mm_training_scaled, Array_training])
                        y_pred_validation = LIRADS_Model.predict([size_mm_validation_scaled, Array_validation])
                        y_pred_testing = LIRADS_Model.predict([size_mm_testing_scaled, Array_testing])
                    
                    cls_pred_training = np.argmax(y_pred_training, axis=1)                   
                    cls_pred_validation = np.argmax(y_pred_validation, axis=1)                  
                    cls_pred_testing = np.argmax(y_pred_testing, axis=1)

                    cls_pred_training = cls_pred_training + 3
                    cls_pred_validation = cls_pred_validation + 3
                    cls_pred_testing = cls_pred_testing + 3

                    # Write to file

                    predict_cls_train = DataFrame({'LIRADS_grade_predicted': cls_pred_training})
                    LIRADS_grade_cls_training_ = DataFrame({'LIRADS_grade':  LIRADS_grade_cls_training})
                    capsule_train = pd.concat([predict_cls_train, LIRADS_grade_cls_training_], axis=1)

                    IDs_training_ = np.array(IDs_training['sample_ID'].values.tolist())
                    IDs_training_df = DataFrame({'sample_ID':  IDs_training_})
                    if ifAug:
                        IDs_training__ = np.array(data_copy(IDs_training_, 7))
                        IDs_training_df = DataFrame({'sample_ID':  IDs_training__})

                    capsule_train = pd.concat([IDs_training_df, capsule_train], axis=1)


                    predict_cls_validation = DataFrame({'LIRADS_grade_predicted': cls_pred_validation})
                    LIRADS_grade_cls_validation_ = DataFrame({'LIRADS_grade': LIRADS_grade_cls_validation})
                    capsule_validation = pd.concat([predict_cls_validation, LIRADS_grade_cls_validation_], axis=1)

                    IDs_validation_ = np.array(IDs_validation['sample_ID'].values.tolist())
                    IDs_validation_df = DataFrame({'sample_ID':  IDs_validation_})

                    capsule_validation = pd.concat([IDs_validation_df, capsule_validation], axis=1)


                    predict_cls_testing = DataFrame({'LIRADS_grade_predicted': cls_pred_testing})
                    LIRADS_grade_cls_testing_ = DataFrame({'LIRADS_grade': LIRADS_grade_cls_testing})
                    capsule_testing = pd.concat([predict_cls_testing, LIRADS_grade_cls_testing_], axis=1)

                    IDs_testing_ = np.array(IDs_testing['sample_ID'].values.tolist())
                    IDs_testing_df = DataFrame({'sample_ID':  IDs_testing_})

                    capsule_testing = pd.concat([IDs_testing_df, capsule_testing], axis=1)


                    # %%
                    # Result output
                    if not home1:
                        LIRADS_grade_result_path = path + '/Result/' + model
                    else:
                        LIRADS_grade_result_path = path + '/' + model

                    path0 = LIRADS_grade_result_path + '/LIRADS_Grade'
                    if not os.path.exists(path0):
                        os.mkdir(path0)

                    path1 = LIRADS_grade_result_path + '/LIRADS_Grade/Group' + str(group)
                    if not os.path.exists(path1):
                        os.mkdir(path1)
                    path2 = path1 + '/5folds-repeat' + str(repeat)
                    if not os.path.exists(path2):
                        os.mkdir(path2)
                    outputpath = path2 + '/' + str(fold)
                    if not os.path.exists(outputpath):
                        os.mkdir(outputpath)
                    if not os.path.exists(outputpath + '/training'):
                        os.mkdir(outputpath + '/training')
                    if not os.path.exists(outputpath + '/validation'):
                        os.mkdir(outputpath + '/validation')
                    if not os.path.exists(outputpath + '/testing'):
                        os.mkdir(outputpath + '/testing')

                    # %%
                    capsule_train.to_csv(outputpath + '/training/liradsGrade', sep=',', index=False, header=True)
                    capsule_validation.to_csv(outputpath + '/validation/liradsGrade', sep=',', index=False, header=True)
                    capsule_testing.to_csv(outputpath + '/testing/liradsGrade', sep=',', index=False, header=True)
