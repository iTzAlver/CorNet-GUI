# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import tensorflow as tf
# from tensorflow import keras
# import random
# from nltk.metrics import windowdiff
# import os
# import numpy as np
# import pickle
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# dbpath = r'../../../db/db/ht/32k_8t_0w_2c.ht'


# -----------------------------------------------------------
def window_diff(a: tf.Tensor, b: tf.Tensor, th=0.5):
    # Data fromatting:
    _b = tf.cast(tf.convert_to_tensor(b), dtype='float32')
    _a = tf.cast(tf.convert_to_tensor(a), dtype='float32')
    _b = tf.where(_b > th, 1, 0)
    _a = tf.where(_a > th, 1, 0)
    _b = tf.cast(_b, dtype='float32')
    _a = tf.cast(_a, dtype='float32')
    access__ = 1
    # Window size:
    # _aux = tf.math.multiply(2.0, (tf.math.add(1.0, tf.math.reduce_sum(_b))))
    # _aux = tf.math.divide(_b.shape[1], _aux)
    # w_size = tf.round(_aux)
    # w_size = tf.cast(w_size, dtype='int32')
    w_size = 3
    # Divider is the number of convolutions.
    _aux = tf.math.subtract(_b.shape[access__], w_size)
    _divider = tf.math.add(_aux, 1)
    # We create the masks of the window.
    _n_shifts = tf.math.subtract(w_size, 1)
    _mask = tf.eye(_divider, num_columns=_b.shape[access__], dtype='float32')
    _masks = tf.eye(_divider, num_columns=_b.shape[access__], dtype='float32')
    while tf.greater(_n_shifts, 0):
        _this_roll = tf.roll(_mask, shift=_n_shifts, axis=1)
        _masks = tf.math.add(_masks, _this_roll)
        _n_shifts = tf.math.subtract(_n_shifts, 1)
    # Addup is the sum of the total error, masked.
    _a_masked = tf.linalg.matvec(_masks, _a)
    _b_masked = tf.linalg.matvec(_masks, _b)
    _aux = tf.subtract(_a_masked, _b_masked)
    _diff_unnorm = tf.math.abs(_aux)
    _aux = tf.where(_diff_unnorm > 0, 1, 0)
    _addup = tf.reduce_sum(_diff_unnorm, axis=access__)
    _addup = tf.cast(_addup, dtype='int32')
    # Return the windowdiff.
    _result = tf.math.divide(_addup, _divider)
    return tf.reduce_mean(_result)
#
#
# def mse(x_, y_):
#     x = tf.cast(tf.convert_to_tensor(x_), dtype='float32')
#     y = tf.cast(tf.convert_to_tensor(y_), dtype='float32')
#     x2 = tf.multiply(x, x)
#     y2 = tf.multiply(y, y)
#     sm = tf.add(x2, y2)
#     rm = tf.reduce_mean(sm)
#     sq = tf.math.sqrt(rm)
#     return sq
#
#
# def test(tests, size):
#     k = 3
#     testings_x = []
#     testings_y = []
#     _testings_x = []
#     _testings_y = []
#     for _ in range(tests):
#         _a_ = [1 if random.randint(0, 3) > 2 else 0 for _ in range(size)]
#         _b_ = [1 if random.randint(0, 3) > 2 else 0 for _ in range(size)]
#         testings_x.append(_a_)
#         testings_y.append(_b_)
#         _testings_x.append([str(x) for x in _a_])
#         _testings_y.append([str(y) for y in _b_])
#     my_calcs = window_diff(tf.convert_to_tensor(testings_x), tf.convert_to_tensor(testings_y))
#
#     real_calcs = []
#     for x, y in zip(_testings_x, _testings_y):
#         real_calcs.append(windowdiff(''.join(x), ''.join(y), k))
#
#     for c1, c2 in zip(my_calcs, real_calcs):
#         assert c1 == c2
#     print(f'No errors detected in {tests} tests. k = {k}')
#
#
# def test2():
#     with open(dbpath, 'rb') as file:
#         db = pickle.load(file)
#     model = keras.Sequential()
#     model.add(keras.layers.Input(shape=(8, 8, 1)))
#     model.add(keras.layers.Conv2D(64, 2))
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dense(180))
#     model.add(keras.layers.Dense(8))
#     model.add(keras.layers.ThresholdedReLU(0.6))
#     xtrain = tf.convert_to_tensor(np.array(db.dataset.xtrain).astype("float32") / 255)
#     ytrain = tf.convert_to_tensor(db.dataset.ytrain)
#     xval = tf.convert_to_tensor(np.array(db.dataset.xval).astype("float32") / 255)
#     yval = tf.convert_to_tensor(db.dataset.yval)
#     model.compile(loss=window_diff, optimizer='adam')
#     inx = xtrain[:3]
#     iny = ytrain[:3]
#     history = model.fit(xtrain, ytrain, batch_size=1, epochs=1, validation_data=(xval, yval))
#     print(history)
#
#
# if __name__ == '__main__':
#     # test(200, 8)
#     test2()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
