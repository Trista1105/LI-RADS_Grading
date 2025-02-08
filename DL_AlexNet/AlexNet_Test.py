# %matplotlib inline
# import matplotlib.pyplot as plt
import shutil

import PIL
import tensorflow as tf
import numpy as np
import os


from keras.layers import Dense, Flatten, Dropout

#################### CLR #################################
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np

#############################################################

def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]


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


from keras import backend as K

K.clear_session()

import keras
from keras.models import load_model

########################################################## AlexNet Model ########################################################################################################
from tensorflow.keras.layers import Conv2D, Lambda, MaxPool2D, Flatten, Dense, Dropout, Activation, \
    ZeroPadding2D, Input
from tensorflow.keras import optimizers, losses, initializers
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

#####################################################################
# model parameters
from keras import optimizers

import matplotlib.pyplot as plt
import numpy as np
import math

import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale

import csv

from pandas.core.frame import DataFrame

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

tf.config.experimental_run_functions_eagerly(True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))



class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


def AlexNet_parallel(input_shape, num_classes):
    # input_shape = (224, 224, 3)
    # num_classes = num_classes
    std = 0.01,  # 0.1
    kernel_initializer = initializers.TruncatedNormal(
        stddev=std)  # initializers.RandomNormal(stddev=std)
    bias_initializer0 = tf.keras.initializers.constant(0)
    bias_initializer1 = tf.keras.initializers.constant(1)
    input = Input(input_shape, name="Input")
    # input size should be : (b x 3 x 227 x 227)
    # The image in the original paper states that width and height are 224 pixels, but
    # the dimensions after first convolution layer do not lead to 55 x 55.
    x = ZeroPadding2D(((3, 0), (3, 0)))(input)

    ################################ conv1 ##################################
    x = Conv2D(96,
               (11, 11),
               4,
               kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer0,
               padding='Same',
               name="conv1")(x)

    x = Activation(activation="relu", name='relu1')(x)
    x = Lambda(tf.nn.local_response_normalization,
               arguments={
                   'alpha': 1e-4,
                   'beta': 0.75,
                   'depth_radius': 5,  # 2, it should be 5 according to Paper
                   'bias': 2.0},
               name="lrn1")(x)
    # keras.layers.core.Lambda(function, output_shape=None, mask=None, arguments=None)
    # arguments：可选，是字典格式，用来传参

    x = MaxPool2D(pool_size=(3, 3),
                  strides=2,
                  padding='VALID',
                  name="pool1")(x)

    ################################### conv2 #################################
    # x = Conv2D(256,
    #            (5, 5),
    #            kernel_initializer=kernel_initializer,
    #            padding="Same",
    #            name="conv2")(x)
    pool1_groups = tf.split(axis=3, value=x, num_or_size_splits=2)

    x_up = Conv2D(128,
                  (5, 5),
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer1,
                  padding="Same",
                  name="conv2_1")(pool1_groups[0])
    x_down = Conv2D(128,
                    (5, 5),
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer1,
                    padding="Same",
                    name="conv2_2")(pool1_groups[1])

    x = tf.concat(axis=3, values=[x_up, x_down])

    x = Activation(activation="relu", name='relu2')(x)
    x = Lambda(tf.nn.local_response_normalization,
               arguments={
                   'alpha': 1e-4,
                   'beta': 0.75,
                   'depth_radius': 5,
                   'bias': 2.0},
               name="lrn2")(x)

    x = MaxPool2D(pool_size=(3, 3),
                  strides=2,
                  padding='VALID',
                  name="pool2")(x)

    ##################################### conv3 ##################################
    x = Conv2D(384,
               (3, 3),
               kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer0,
               padding="Same",
               name="conv3")(x)

    x = Activation(activation="relu", name='relu3')(x)
    ##################################### conv4 ##################################
    # x = Conv2D(384,
    #            (3, 3),
    #            kernel_initializer=kernel_initializer,
    #            padding="Same",
    #            name="conv4")(x)

    relu3_groups = tf.split(axis=3, value=x, num_or_size_splits=2)

    x_up = Conv2D(192,
                  (3, 3),
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer1,
                  padding="Same",
                  name="conv4_1")(relu3_groups[0])
    x_down = Conv2D(192,
                    (3, 3),
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer1,
                    padding="Same",
                    name="conv4_2")(relu3_groups[1])

    x = tf.concat(axis=3, values=[x_up, x_down])

    x = Activation(activation="relu", name='relu4')(x)
    ##################################### conv5 ##################################
    # x = Conv2D(256,
    #            (3, 3),
    #            kernel_initializer=kernel_initializer,
    #            padding="Same",
    #            name="conv5")(x)

    relu4_groups = tf.split(axis=3, value=x, num_or_size_splits=2)

    x_up = Conv2D(128,
                  (3, 3),
                  kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer1,
                  padding="Same",
                  name="conv5_1")(relu4_groups[0])
    x_down = Conv2D(128,
                    (3, 3),
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer1,
                    padding="Same",
                    name="conv5_2")(relu4_groups[1])

    x = tf.concat(axis=3, values=[x_up, x_down])

    x = Activation(activation="relu", name='relu5')(x)

    x = MaxPool2D(pool_size=(3, 3),
                  strides=2,
                  padding='VALID',
                  name="pool5")(x)
    ##################################### Flatten ##################################
    x = Flatten(name="flattened6")(x)
    ##################################### fc6 ##################################
    x = Dense(1024,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer0,
              name="fc6")(x)
    x = Activation(activation="relu", name='relu6')(x)
    x = Dropout(0.5, name="dropout6")(x)
    ##################################### fc7 ##################################
    x = Dense(512,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer0,
              name="fc7")(x)
    x = Activation(activation="relu", name='relu7')(x)
    x = Dropout(0.5, name="dropout7")(x)

    ##################################### fc8 ##################################
    x = keras.layers.Dense(num_classes,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer0,
                           name="fc8")(x)
    output = Activation(activation="softmax", name='softmax8')(x)
    model = keras.Model(inputs=input, outputs=output, name="AlexNet")

    return model


def get_lr_metric(optimizer):  # printing the value of the learning rate
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr

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

def data_testing(Array, Index):
    Array_sub = []
    for i in range(0, len(Index)):
        index = Index[i]
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



################################ Model #####################################

num_classes = 3

model = 'AlexNet'
# learning rate
# VGG16 Paper: 4e-5~4e-4, decay=0.9, cycle length=3, factor*cycle length, factor = 2
# AlexNet Paper: 1e-5~4e-2, clinical learning rate, triangular waveform
if model == 'VGG16':
    learning_rate = 1e-4
if model == 'AlexNet':
    learning_rate = 1e-4

ifAug = True

img_input = 'patches_array_aug'

data_site = ''



path = '/' # root path of the model
modelpath = path + '/Result/AlexNet'

predict_path = path + '/Predict/AlexNet/' + data_site
if not os.path.exists(predict_path):
    os.mkdir(predict_path)

# AlexNet
if __name__ == '__main__':

    # 1. Read LIRADS Grade Label
    Labels = pd.read_excel(path + '/Data/.xlsx', sheet_name='')

    Labels = pd.concat([Labels[['Sample_ID']], Labels[['size_mm']], Labels[['APHE']], Labels[['Washout']],
                        Labels[['Capsule_enhancement']], Labels[['LIRADS_grade']]], axis=1)
    size_mm = pd.concat([Labels[['Sample_ID']], Labels[['size_mm']]], axis=1)
    LIRADS_grade_true = pd.concat([Labels[['Sample_ID']], Labels[['LIRADS_grade']]], axis=1)


    Labels_external = pd.read_excel(path + '/Data/.xlsx', sheet_name='')


    Labels_external = pd.concat([Labels_external[['Sample_ID']], Labels_external[['size_mm']], Labels_external[['APHE']], Labels_external[['Washout']],
                        Labels_external[['Capsule_enhancement']], Labels_external[['LIRADS_grade']]], axis=1)
    size_mm_external = pd.concat([Labels_external[['Sample_ID']], Labels_external[['size_mm']]], axis=1)
    LIRADS_grade_true_external = pd.concat([Labels_external[['Sample_ID']], Labels_external[['LIRADS_grade']]], axis=1)

    # 2. Read IDs and ID Index for all the data
    IDs = pd.read_csv(path + '/Data/IDs.csv', header=0)

    IDs_external = pd.read_csv(path + '/Data_' + data_site + '/IDs.csv', header=0)

    # 5. Read image pixel value data

    if img_input == 'img_array':
        Pre_Array = np.load(path + '/Data/Pre_Array.npy')
        A_Array = np.load(path + '/Data/A_Array.npy')
        V_Array = np.load(path + '/Data/V_Array.npy')
        D_Array = np.load(path + '/Data/D_Array.npy')
    if img_input == 'img_array_aug':
        Pre_Array = np.load(path + '/Data/Pre_Array_Aug.npy')
        A_Array = np.load(path + '/Data/A_Array_Aug.npy')
        V_Array = np.load(path + '/Data/V_Array_Aug.npy')
        D_Array = np.load(path + '/Data/D_Array_Aug.npy')
    if img_input == 'patches_array_aug':
        Pre_Array = np.load(path + '/Data/Pre_Array_Patches_Scaled_Aug.npy')
        A_Array = np.load(path + '/Data/A_Array_Patches_Scaled_Aug.npy')
        V_Array = np.load(path + '/Data/V_Array_Patches_Scaled_Aug.npy')
        D_Array = np.load(path + '/Data/D_Array_Patches_Scaled_Aug.npy')

    # Array_Pre_A_V_D = np.concatenate([V_Array, D_Array, A_Array],axis=1)
    Array_Pre_A_V_D = np.zeros((A_Array.shape[0], 224, 224, 3))
    Array_Pre_A_V_D[:, :, :, 0] = V_Array
    Array_Pre_A_V_D[:, :, :, 1] = D_Array
    Array_Pre_A_V_D[:, :, :, 2] = A_Array



    Array_Pre_A_V_D = np.zeros((A_Array.shape[0], 224, 224, 3))
    Array_Pre_A_V_D[:, :, :, 0] = V_Array
    Array_Pre_A_V_D[:, :, :, 1] = D_Array
    Array_Pre_A_V_D[:, :, :, 2] = A_Array


    adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  # 0.001
    lr_metric = get_lr_metric(optimizer=adam)


    for group in range(1, 6):
        for repeat in range(1, 6):
            for fold in range(1, 6):

                print('group: ', group, ', repeat: ', repeat, ', fold: ', fold)


                # data path for training, validation and testing
                subpath = path + '/Data/Groups/Group' + str(group) + '/5folds-repeat' + str(repeat) + '/' + str(fold)

                IDs_training = pd.read_csv(subpath + '/training/liradsGrade', header=0)
                IDs_training = pd.concat([IDs_training[['sample_ID']], IDs_training[['LIRADS_grade']]], axis=1)

                IDs_validation = pd.read_csv(subpath + '/validation/liradsGrade', header=0)
                IDs_testing = pd.read_csv(subpath + '/testing/liradsGrade', header=0)

                IDs_validation = pd.concat([IDs_validation[['sample_ID']], IDs_validation[['LIRADS_grade']]], axis=1)
                IDs_testing = pd.concat([IDs_testing[['sample_ID']], IDs_testing[['LIRADS_grade']]], axis=1)


                # 4. Add ID Index to training, validation and testing
                IDs_training_with_index = pd.merge(left=IDs_training, right=IDs, how='left', left_on='sample_ID',
                                                   right_on='sample_ID')
                IDs_validation_with_index = pd.merge(left=IDs_validation, right=IDs, how='left', left_on='sample_ID',
                                                     right_on='sample_ID')
                IDs_testing_with_index = pd.merge(left=IDs_testing, right=IDs, how='left', left_on='sample_ID',
                                                  right_on='sample_ID')

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

                mean_training = np.mean(IDs_training_size_mm_)
                std_training = np.std(IDs_training_size_mm_)

                if data_site == '':

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

                    size_mm_validation_scaled = (IDs_validation_size_mm_ - mean_training) / std_training
                    size_mm_testing_scaled = (IDs_testing_size_mm_ - mean_training) / std_training

                else:
                    IDs_testing_with_index = pd.merge(left=LIRADS_grade_true_external, right=IDs_external, how='left', left_on='Sample_ID',
                                                      right_on='sample_ID')

                    testing_index = IDs_testing_with_index['Index']
                    testing_index_ = testing_index.values.tolist()

                    Array_testing = np.array(data_testing(Array_Pre_A_V_D, testing_index_))

                    LIRADS_grade_cls_testing = np.array(LIRADS_grade_true_external['LIRADS_grade'].values.tolist())
                    LIRADS_grade_testing = to_categorical(LIRADS_grade_cls_testing - 3)

                    size_mm_testing = np.array(size_mm_external['size_mm'].values.tolist())
                    # size_mm_testing_scaled = scale(size_mm_testing)
                    size_mm_testing_scaled = (size_mm_testing - mean_training) / std_training

                modelsubpath = modelpath + '/Group' + str(group) + '/5folds-repeat' + str(repeat) + '/' + str(fold)

                # Load best model and do prediction
                if os.path.exists(modelsubpath + '/AlexNet/log/checkpoint/best_weights.hdf5'):
                    AlexNet_model = load_model(modelsubpath + '/AlexNet/log/checkpoint/best_weights.hdf5', custom_objects = {"lr": lr_metric})


                    AlexNet_model.compile(optimizer=adam, loss='categorical_crossentropy',
                                          metrics=['categorical_accuracy', lr_metric])


                    # scores_training = AlexNet_model.evaluate(Array_training, LIRADS_grade_training, verbose=1)
                    # scores_validation = AlexNet_model.evaluate(Array_validation, LIRADS_grade_validation, verbose=1)
                    # scores_testing = AlexNet_model.evaluate(Array_testing, LIRADS_grade_testing, verbose=1)
                    #
                    # print(scores_training, scores_validation, scores_testing)

                    # print('group: ', group, ', repeat: ', repeat, ', fold: ', fold)


                    ################################################################################
                    # Prediction

                    if data_site == '':
                        y_pred_training = AlexNet_model.predict(Array_training)
                        cls_pred_training = np.argmax(y_pred_training, axis=1)

                        y_pred_validation = AlexNet_model.predict(Array_validation)
                        cls_pred_validation = np.argmax(y_pred_validation, axis=1)

                        cls_pred_training = cls_pred_training + 3
                        cls_pred_validation = cls_pred_validation + 3

                    y_pred_testing = AlexNet_model.predict(Array_testing)
                    cls_pred_testing = np.argmax(y_pred_testing, axis=1)

                    cls_pred_testing = cls_pred_testing + 3

                    # Write to file

                    if data_site == '':

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

                    if data_site == '':
                        IDs_testing_ = np.array(IDs_testing['sample_ID'].values.tolist())
                    else:
                        IDs_testing_ = np.array(Labels_external['Sample_ID'].values.tolist())
                    IDs_testing_df = DataFrame({'sample_ID':  IDs_testing_})

                    capsule_testing = pd.concat([IDs_testing_df, capsule_testing], axis=1)


                    # %%
                    # Result output
                    import os
                    path1 = predict_path + '/Group' + str(group)
                    if not os.path.exists(path1):
                        os.mkdir(path1)
                    path2 = path1 + '/5folds-repeat' + str(repeat)
                    if not os.path.exists(path2):
                        os.mkdir(path2)

                    outputpath = path2 + '/' + str(fold)
                    if not os.path.exists(outputpath):
                        os.mkdir(outputpath)
                    if data_site == '':
                        if not os.path.exists(outputpath + '/training'):
                            os.mkdir(outputpath + '/training')
                        if not os.path.exists(outputpath + '/validation'):
                            os.mkdir(outputpath + '/validation')
                        if not os.path.exists(outputpath + '/testing'):
                            os.mkdir(outputpath + '/testing')

                        capsule_train.to_csv(outputpath + '/training/liradsGrade', sep=',', index=False, header=True)
                        capsule_validation.to_csv(outputpath + '/validation/liradsGrade', sep=',', index=False, header=True)
                        capsule_testing.to_csv(outputpath + '/testing/liradsGrade', sep=',', index=False, header=True)
                    else:
                        capsule_testing.to_csv(outputpath + '/liradsGrade', sep=',', index=False, header=True)

                else:
                    print(modelsubpath + '/AlexNet/log/checkpoint/best_weights.hdf5', 'Not Exist!!')