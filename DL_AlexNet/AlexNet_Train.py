# Auto judge if need to stop


# %matplotlib inline
# import matplotlib.pyplot as plt
import shutil

import PIL
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
import keras
print("Keras version:", keras.__version__)
import numpy as np
import os

from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop


onServer = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
#################### CLR #################################
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np


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


#############################################################

def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]


import sklearn
# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix



def print_confusion_matrix(cls_pred, cls_test, class_names):
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


from keras import backend as K



######################################################### AlexNet Model ########################################################################################################
from tensorflow.keras.layers import Conv2D, Lambda, MaxPool2D, Flatten, Dense, Dropout, Activation, \
    ZeroPadding2D, Input
from tensorflow.keras import optimizers, losses, initializers
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import keras
if not onServer:
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
from sklearn.utils import class_weight
from pandas.core.frame import DataFrame
import os

K.clear_session()


from keras.models import load_model

import datetime


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



# class_weight = {0:3,1:3,2:1}
                # class_weight = {0:1,1:1,2:1}

# 1. learning rate 1: constant
# sgd = optimizers.gradient_descent_v2.SGD(lr=0.01, decay = 0, momentum=0, nesterov=False)
# adam = Adam(lr=learning_rate)  # 1e-5, 1e-6

# 2. learning rate 2: time based decay
if onServer:
    sgd = optimizers.SGD(lr=1e-5, decay=0.0005, momentum=0.9, nesterov=False)  # 1e-5, 1e-6
else:
    # sgd = optimizers.gradient_descent_v2.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)
    sgd = optimizers.gradient_descent_v2.SGD(lr=1e-5, decay=0.0005, momentum=0.9, nesterov=False)

# LearningRate = LearningRate * 1/(1 + decay * epoch)

# 3. learning rate 3: step decay
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,
                                     math.floor((1 + epoch) / epochs_drop))
    return lrate


# lr = lr0 * drop^floor(epoch / epochs_drop)

def temp(model):
    def scheduler(epoch):
        lr = K.get_value(model.optimizer.lr)

        if epoch == 13:
            lr *= 0.5
        elif epoch == 11:
            lr *= 0.1
        elif epoch == 9:
            lr *= 0.1
        elif epoch == 5:
            lr *= 0.1
        print(lr)
        return lr

    return scheduler


# 4. learning rate 4: Exponential Decay
def exp_decay(epoch):
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * math.exp(-k * epoch)
    return lrate


# lr = lr0 * e ^（-kt）

lrate = LearningRateScheduler(step_decay)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))

def get_lr_metric(optimizer):  # printing the value of the learning rate
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


loss_history = LossHistory()




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


def print_layer_trainable_AlexNet():
    for layer in AlexNet_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

model = 'AlexNet'
home1 = 'True'



path = '/' # root path to save AlexNet trained model
modeloutpath = path + '/Result/AlexNet'
xlspath = path + '/Result/' + model + '/LIRADS-AlexNet.csv'


WEIGHTS_PATH = '/.../AlexNet' # path of AlexNet Weights




if not os.path.exists(modeloutpath):
    os.mkdir(modeloutpath)


name_list = ['Group', 'Repeat', 'Fold', 'Training Loss', 'Validation Loss', 'Testing Loss',
             'Training Accuracy', 'Validation Accuracy', 'Testing Accuracy', 'Best Epoch']
create_csv_with_header(xlspath, name_list)

################################ Model #####################################
ifAug = True

num_classes = 3

#####################################################################
# model parameters


# learning rate
# VGG16 Paper: 4e-5~4e-4, decay=0.9, cycle length=3, factor*cycle length, factor = 2
# AlexNet Paper: 1e-5~4e-2, clinical learning rate, triangular waveform
if onServer:
    if model == 'VGG16':
        learning_rate = 1e-4
        batch_size = 128  # VGG16 Paper:32,  AlexNet Paper:8
    if model == 'AlexNet':
        learning_rate = 1e-4
        batch_size = 128  # tested: 128
else:
    learning_rate = 1e-4
    batch_size = 1  # VGG16 Paper:32,  AlexNet Paper:8
# VGG16 Paper: SGD with momentum, SGD with restarts


epochs = 250 # 250  # VGG16 Paper: 400+400, AlexNet Paper: 200, stop at 53

patience = 20 # 自定义的patience=15目前没有使用，Early Stopping中用的patience是20

loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']


# AlexNet
if __name__ == '__main__':
    for group in range(1, 2):
        for repeat in range(5, 6):
            for fold in range(1, 6):
                # 5. Adaptive learning rate
                if onServer:
                    # keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
                    # keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
                    # keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
                    # adam = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)  # 0.001
                    adam = keras.optimizers.adam_v2.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                         decay=0.0)  # AttributeError: module 'keras.optimizers' has no attribute 'adam_v2'

                else:
                    adam = optimizers.adam_v2.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

                # 6. ReduceLROnPlateau
                reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=0,
                                                              mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-6)

                # 1. Read LIRADS Grade Label

                Labels = pd.read_excel(path + '/...xlsx', sheet_name='')
                Labels = pd.concat(
                    [Labels[['Sample_ID']], Labels[['size_mm']], Labels[['APHE']], Labels[['Washout']],
                     Labels[['Capsule_enhancement']], Labels[['LIRADS_grade']]], axis=1)

                size_mm = pd.concat([Labels[['Sample_ID']], Labels[['size_mm']]], axis=1)
                LIRADS_grade_true = pd.concat([Labels[['Sample_ID']], Labels[['LIRADS_grade']]], axis=1)

                # 2. Read IDs and ID Index for all the data
                IDs = pd.read_csv(path + '/.../IDs.csv', header=0)

                # 3. Read IDs and Labels for training, validation and testing

                print('group: ', group, ', repeat: ', repeat, ', fold: ', fold)


                subpath = path + '/Data/Groups/Group' + str(group) + '/5folds-repeat' + str(repeat) + '/' + str(fold)  # read the existing data split


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


                Pre_Array = np.load(path + '/Data/Pre_Array_Patches_Scaled_Aug.npy')
                A_Array = np.load(path + '/Data/A_Array_Patches_Scaled_Aug.npy')
                V_Array = np.load(path + '/Data/V_Array_Patches_Scaled_Aug.npy')
                D_Array = np.load(path + '/Data/D_Array_Patches_Scaled_Aug.npy')


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

                # # Calculate class weight
                # # 计算类别权重
                # my_class_weight = class_weight.compute_class_weight(class_weight='balanced',  classes=np.unique(LIRADS_grade_cls_training),y=LIRADS_grade_cls_training)
                # # 需要转成字典
                # class_weight_dict = dict(enumerate(my_class_weight))
                #
                # print(class_weight_dict)

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
                    size_mm_training_scaled = scale(IDs_training_size_mm__) # scale: Center to the mean and component wise scale to unit variance.

                else:
                    size_mm_training_scaled = scale(IDs_training_size_mm_)

                mean_training = np.mean(IDs_training_size_mm_)
                std_training = np.std(IDs_training_size_mm_)

                size_mm_validation_scaled = (IDs_validation_size_mm_ - mean_training) / std_training
                size_mm_testing_scaled = (IDs_testing_size_mm_ - mean_training) / std_training


                outpath = modeloutpath + '/Group' + str(group) + '/5folds-repeat' + str(repeat) + '/' + str(fold)

                ############################################# Model ############################################################################################
                keras.backend.clear_session()


                # 获取当前时间戳
                def get_timestamp():
                    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


                # 设定路径
                outpath = path + '/AlexNet/Group' + str(group) + '/5folds-repeat' + str(repeat) + '/' + str(fold)
                model_subpath = ''  # 如果有子路径，请在这里设置
                checkpoint_dir = os.path.join(outpath, model_subpath, 'checkpoint')

                # 确保检查点目录存在
                os.makedirs(checkpoint_dir, exist_ok=True)


                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                if os.path.exists(outpath + model_subpath + f'/checkpoint/best_weights_{timestamp}_fold{fold}_epoch{{epoch:02d}}-acc{{categorical_accuracy}}-val_acc{{val_categorical_accuracy}}.hdf5'):
                    os.remove(outpath + model_subpath + f'/checkpoint/best_weights_{timestamp}_fold{fold}_epoch{{epoch:02d}}-acc{{categorical_accuracy}}-val_acc{{val_categorical_accuracy}}.hdf5')

                checkpoint1 = keras.callbacks.ModelCheckpoint(
                    filepath=outpath + model_subpath + f'/checkpoint/best_weights_{timestamp}_fold{fold}_epoch{{epoch:02d}}-acc{{categorical_accuracy}}-val_acc{{val_categorical_accuracy}}.hdf5',
                    # filepath=outpath + model_subpath + '/checkpoint/best_weights_epoch-{epoch:02d}-acc-{categorical_accuracy}-val_acc-{val_categorical_accuracy}.hdf5',
                    monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                    save_weights_only=True, mode='auto', period=1) # save_weights_only=False  ValueError: Unable to create dataset (name already exists)

                tensorboard = keras.callbacks.TensorBoard(
                    log_dir=outpath + model_subpath + '/tensorboard',
                    write_graph=True,
                    write_images=True
                    #         histogram_freq=1,
                    #         embeddings_freq=1,
                    #         embeddings_data=x_train[:1000].astype('float32')
                )
                earlystopping = EarlyStopping(monitor='val_categorical_accuracy', patience=patience, verbose=2, mode='max') # patience=20
                # clr = CyclicLR(base_lr=5e-4, max_lr=5e-2, step_size=130, mode='triangular') # 1e-5~5e-4

                callbacks = [checkpoint1, tensorboard, reduce_lr, earlystopping]


                AlexNet_model = AlexNet_parallel(input_shape=(224, 224, 3), num_classes=num_classes)
                AlexNet_model.summary()

                transferLearning = True

                if transferLearning:
                    # Get model weights
                    # alexnet_model_weights = AlexNet_model.get_weights()

                    # Set model weights
                    weights_dict = np.load(os.path.join(WEIGHTS_PATH, 'bvlc_alexnet.npy'), encoding='bytes',
                                           allow_pickle=True).item()

                    # AlexNet_model.set_weights(weights_dict) #dimention doesn't match

                    # Get layer weights
                    # conv1_weights = AlexNet_model.get_layer('conv1').get_weights()

                    # Set layer weights
                    for key in weights_dict:
                        if 'conv1' in str(key):
                            conv1 = weights_dict[key]
                        if 'conv2' in str(key):
                            conv2 = weights_dict[key]
                        if 'conv3' in str(key):
                            conv3 = weights_dict[key]
                        if 'conv4' in str(key):
                            conv4 = weights_dict[key]
                        if 'conv5' in str(key):
                            conv5 = weights_dict[key]
                        if 'fc6' in str(key):
                            fc6 = weights_dict[key]
                        if 'fc7' in str(key):
                            fc7 = weights_dict[key]
                        if 'fc8' in str(key):
                            fc8 = weights_dict[key]

                    convs = [conv1, conv2, conv3, conv4, conv5]

                    for i in range(1, 6):

                        if i == 2 or i == 4 or i == 5:

                            convs_up_down = convs[i - 1]
                            convs_up_down_weights = convs_up_down[0]
                            convs_up_down_biases = convs_up_down[1]

                            for j in range(1, 3):
                                layer_name = 'conv' + str(i) + '_' + str(j)
                                convs_weights = convs_up_down_weights[:, :, :,
                                                int((j - 1) * convs_up_down_weights.shape[3] / 2):int(
                                                    j * convs_up_down_weights.shape[3] / 2)]
                                convs_biases = convs_up_down_biases[
                                               int((j - 1) * convs_up_down_weights.shape[3] / 2):int(
                                                   j * convs_up_down_weights.shape[3] / 2)]

                                AlexNet_model.get_layer(layer_name).set_weights([convs_weights, convs_biases])
                        else:

                            layer_name = 'conv' + str(i)
                            AlexNet_model.get_layer(layer_name).set_weights(convs[i - 1])





                # plot_model(AlexNet_model) #must `pip install pydot` and install graphviz(https://graphviz.gitlab.io/download/), sudo apt install graphviz

                # print_layer_trainable_AlexNet()
                #
                # for layer in AlexNet_model.layers:
                #     layer.trainable = False
                #
                # for layer in AlexNet_model.layers:
                #     # Boolean whether this layer is trainable.
                #     # trainable = ('block5' in layer.name or 'block4' in layer.name or 'flatten' in layer.name)
                #     trainable = ('conv' not in layer.name)
                #
                #     # Set the layer's bool.
                #     layer.trainable = trainable
                #
                #
                #
                # print_layer_trainable_AlexNet()

                # AlexNet_model.compile(optimizer=adam, loss=loss, metrics=metrics)
                lr_metric = get_lr_metric(optimizer=adam)
                AlexNet_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy', lr_metric])

                ############################################# Model ############################################################################################


                bestValAcc = 0
                waite_epoch = 0
                last_10_val_loss = []
                best_epoch = 0

                for i in range(0, 1):
                    print('epoch: ', str(i+1))

                    # for i in range(len(AlexNet_model.weights)):
                    #     AlexNet_model.weights[i]._handle_name = AlexNet_model.weights[i].name + "_" + str(i)
                    # for i, w in enumerate(AlexNet_model.weights): print(i, w.name)
                    #
                    # for i in range(len(AlexNet_model.layers)): AlexNet_model.layers[i]._handle_name = AlexNet_model.layers[i]._name + "_" + str(i)
                    # for i, w in enumerate(AlexNet_model.weights): print(i, w.name)

                    history = AlexNet_model.fit(Array_training, LIRADS_grade_training, \
                                               validation_data = (Array_validation, LIRADS_grade_validation), \
                                               batch_size=batch_size,epochs=epochs, shuffle=True, verbose=1, callbacks=callbacks)



                AlexNet_model.save(outpath + model_subpath + '/alexnet_final_epoch')


                ######################################################## Save best model ##########################################
                files = os.listdir(outpath + model_subpath + '/checkpoint')
                files.sort(key=lambda x: os.path.getmtime((outpath + model_subpath + '/checkpoint' + "/" + x))) # 对checkpoint文件夹下的文件根据时间进行排序
                shutil.copyfile(outpath + model_subpath + '/checkpoint/' + files[-1], outpath + model_subpath + '/best_weights.hdf5') # 排序后，最后一个保存的是best model.
                print('Best model saved!')

                ######################################################### End Save best model ##########################################

                ########################################################################################################################
                ########################################################## Load best model and do prediction ###########################
                ########################################################################################################################

                if os.path.exists(outpath + model_subpath + '/best_weights.hdf5'):
                    # AlexNet_model = load_model(outpath + model_subpath + '/best_weights.hdf5',
                    #                            custom_objects={"lr": lr_metric})
                    AlexNet_model.load_weights(outpath + model_subpath + '/best_weights.hdf5')

                # if os.path.exists(outpath + '/AlexNet/log/alexnet_bestValAcc_model.h'):
                #     AlexNet_model = load_model(outpath + '/AlexNet/log/alexnet_bestValAcc_model.h', custom_objects = {"lr": lr_metric})

                # if os.path.exists(outpath + '/AlexNet/log/alexnet_final_epoch.h'):
                #     AlexNet_model = load_model(outpath + '/AlexNet/log/alexnet_final_epoch.h', custom_objects = {"lr": lr_metric})



                    AlexNet_model.compile(optimizer=adam, loss='categorical_crossentropy',
                                          metrics=['categorical_accuracy', lr_metric])


                    scores_training = AlexNet_model.evaluate(Array_training, LIRADS_grade_training, verbose=1)
                    scores_validation = AlexNet_model.evaluate(Array_validation, LIRADS_grade_validation, verbose=1)
                    scores_testing = AlexNet_model.evaluate(Array_testing, LIRADS_grade_testing, verbose=1)

                    print(scores_training, scores_validation, scores_testing)

                    print('group: ', group, ', repeat: ', repeat, ', fold: ', fold)

                    result_list = [group, repeat, fold, scores_training[0], scores_validation[0], scores_testing[0], scores_training[1], scores_validation[1], scores_testing[1], best_epoch]
                    append_to_csv(xlspath, result_list)

                ################################################################################
                    # Prediction

                    y_pred_training = AlexNet_model.predict(Array_training)
                    cls_pred_training = np.argmax(y_pred_training, axis=1)

                    y_pred_validation = AlexNet_model.predict(Array_validation)
                    cls_pred_validation = np.argmax(y_pred_validation, axis=1)

                    y_pred_testing = AlexNet_model.predict(Array_testing)
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