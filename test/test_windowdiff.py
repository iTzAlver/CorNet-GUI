# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
from tensorflow import keras
import random
from nltk.metrics import windowdiff
import os
import numpy as np
import pickle
from src.compiler_structures.loss_functions import window_diff
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dbpath = r'../../../db/db/ht/32k_8t_0w_2c.ht'


# -----------------------------------------------------------


def mse(x_, y_):
    x = tf.cast(tf.convert_to_tensor(x_), dtype='float32')
    y = tf.cast(tf.convert_to_tensor(y_), dtype='float32')
    x2 = tf.multiply(x, x)
    y2 = tf.multiply(y, y)
    sm = tf.add(x2, y2)
    rm = tf.reduce_mean(sm)
    sq = tf.math.sqrt(rm)
    return sq


def test(tests, size):
    k = 3
    testings_x = []
    testings_y = []
    _testings_x = []
    _testings_y = []
    for _ in range(tests):
        _a_ = [1 if random.randint(0, 3) > 2 else 0 for _ in range(size)]
        _b_ = [1 if random.randint(0, 3) > 2 else 0 for _ in range(size)]
        testings_x.append(_a_)
        testings_y.append(_b_)
        _testings_x.append([str(x) for x in _a_])
        _testings_y.append([str(y) for y in _b_])
    my_calcs = window_diff(tf.convert_to_tensor(testings_x), tf.convert_to_tensor(testings_y))

    real_calcs = []
    for x, y in zip(_testings_x, _testings_y):
        real_calcs.append(windowdiff(''.join(x), ''.join(y), k))

    for c1, c2 in zip(my_calcs, real_calcs):
        assert c1 == c2
    print(f'No errors detected in {tests} tests. k = {k}')


def test2():
    with open(dbpath, 'rb') as file:
        db = pickle.load(file)
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(8, 8, 1)))
    model.add(keras.layers.Conv2D(64, 2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(180))
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.ThresholdedReLU(0.6))
    xtrain = tf.convert_to_tensor(np.array(db.dataset.xtrain).astype("float32") / 255)
    ytrain = tf.convert_to_tensor(db.dataset.ytrain)
    xval = tf.convert_to_tensor(np.array(db.dataset.xval).astype("float32") / 255)
    yval = tf.convert_to_tensor(db.dataset.yval)
    model.compile(loss=window_diff, optimizer='adam')
    inx = xtrain[:3]
    iny = ytrain[:3]
    history = model.fit(xtrain, ytrain, batch_size=1, epochs=1, validation_data=(xval, yval))
    print(history)


if __name__ == '__main__':
    test(200, 8)
    test2()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
