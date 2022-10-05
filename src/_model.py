# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import random
import sys
from tensorflow import keras, convert_to_tensor, distribute
from keras.utils.vis_utils import plot_model

DRAW_MODEL_PATH = r'../multimedia/render'
MODEL_LOCATION = r'../db/models'
MODEL_TEMP = r'../temp/tempmodel.h5'


class Logger:
    stdout = sys.stdout
    messages = []

    def start(self):
        sys.stdout = self

    def stop(self):
        sys.stdout = self.stdout

    def write(self, text):
        self.messages.append(text)

    def flush(self):
        pass


# -----------------------------------------------------------
class Model:
    def __init__(self, compiler, print_=True):
        self.model = None
        self.summary = None
        self._print = print_
        self.compiler = compiler
        self.devices = compiler.devices
        self.compile()
        self.history = None
        self.scope = None

    def compile(self):
        compiler = self.compiler
        self.scope = self._model_scope(self.devices)

        with self.scope:
            model = keras.Sequential(name='compiled-model-keras')
            model.add(keras.Input(shape=(compiler.tput, compiler.tput, 1)))

            for layer_type, layer_shape, layer_args in zip(compiler.layers, compiler.shapes, compiler.args):
                # Core layers:
                if layer_type == 'Dense':
                    model.add(keras.layers.Dense(*layer_shape, **layer_args))
                elif layer_type == 'Flatten':
                    model.add(keras.layers.Flatten(*layer_shape, **layer_args))
                elif layer_type == 'Activation':
                    model.add(keras.layers.Activation(*layer_shape, **layer_args))
                elif layer_type == 'Embedding':
                    model.add(keras.layers.Embedding(*layer_shape, **layer_args))
                elif layer_type == 'Masking':
                    model.add(keras.layers.Masking(*layer_shape, **layer_args))
                elif layer_type == 'Lambda':
                    model.add(keras.layers.Lambda(*layer_shape, **layer_args))
                # Convolutional layers:
                elif layer_type == 'Conv1D':
                    model.add(keras.layers.Conv1D(*layer_shape, **layer_args))
                elif layer_type == 'Conv2D':
                    model.add(keras.layers.Conv2D(*layer_shape, **layer_args))
                elif layer_type == 'Conv3D':
                    model.add(keras.layers.Conv3D(*layer_shape, **layer_args))
                elif layer_type == 'SeparableConv1D':
                    model.add(keras.layers.SeparableConv1D(*layer_shape, **layer_args))
                elif layer_type == 'SeparableConv2D':
                    model.add(keras.layers.SeparableConv2D(*layer_shape, **layer_args))
                elif layer_type == 'DepthwiseConv2D':
                    model.add(keras.layers.DepthwiseConv2D(*layer_shape, **layer_args))
                elif layer_type == 'Conv1DTranspose':
                    model.add(keras.layers.Conv1DTranspose(*layer_shape, **layer_args))
                elif layer_type == 'Conv2DTranspose':
                    model.add(keras.layers.Conv2DTranspose(*layer_shape, **layer_args))
                elif layer_type == 'Conv3DTranspose':
                    model.add(keras.layers.Conv3DTranspose(*layer_shape, **layer_args))
                # Pooling layers:
                elif layer_type == 'MaxPooling1D':
                    model.add(keras.layers.MaxPooling1D(*layer_shape, **layer_args))
                elif layer_type == 'MaxPooling2D':
                    model.add(keras.layers.MaxPooling2D(*layer_shape, **layer_args))
                elif layer_type == 'MaxPooling3D':
                    model.add(keras.layers.MaxPooling3D(*layer_shape, **layer_args))
                elif layer_type == 'AveragePooling1D':
                    model.add(keras.layers.AveragePooling1D(*layer_shape, **layer_args))
                elif layer_type == 'AveragePooling2D':
                    model.add(keras.layers.AveragePooling2D(*layer_shape, **layer_args))
                elif layer_type == 'AveragePooling3D':
                    model.add(keras.layers.AveragePooling3D(*layer_shape, **layer_args))
                elif layer_type == 'GlobalMaxPooling1D':
                    model.add(keras.layers.GlobalMaxPooling1D(*layer_shape, **layer_args))
                elif layer_type == 'GlobalMaxPooling2D':
                    model.add(keras.layers.GlobalMaxPooling2D(*layer_shape, **layer_args))
                elif layer_type == 'GlobalMaxPooling3D':
                    model.add(keras.layers.GlobalMaxPooling3D(*layer_shape, **layer_args))
                elif layer_type == 'GlobalAveragePooling1D':
                    model.add(keras.layers.GlobalAveragePooling1D(*layer_shape, **layer_args))
                elif layer_type == 'GlobalAveragePooling2D':
                    model.add(keras.layers.GlobalAveragePooling2D(*layer_shape, **layer_args))
                elif layer_type == 'GlobalAveragePooling3D':
                    model.add(keras.layers.GlobalAveragePooling3D(*layer_shape, **layer_args))
                # Recursant layers.
                elif layer_type == 'LSTM':
                    model.add(keras.layers.LSTM(*layer_shape, **layer_args))
                elif layer_type == 'GRU':
                    model.add(keras.layers.GRU(*layer_shape, **layer_args))
                elif layer_type == 'SimpleRNN':
                    model.add(keras.layers.SimpleRNN(*layer_shape, **layer_args))
                elif layer_type == 'RNN':
                    model.add(keras.layers.RNN(*layer_shape, **layer_args))
                elif layer_type == 'TimeDistributed':
                    model.add(keras.layers.TimeDistributed(*layer_shape, **layer_args))
                elif layer_type == 'Bidirectional':
                    model.add(keras.layers.Bidirectional(*layer_shape, **layer_args))
                elif layer_type == 'ConvLSTM1D':
                    model.add(keras.layers.ConvLSTM1D(*layer_shape, **layer_args))
                elif layer_type == 'ConvLSTM2D':
                    model.add(keras.layers.ConvLSTM2D(*layer_shape, **layer_args))
                elif layer_type == 'ConvLSTM3D':
                    model.add(keras.layers.ConvLSTM3D(*layer_shape, **layer_args))
                # Preprocessing layers:
                elif layer_type == 'TextVectorization':
                    model.add(keras.layers.TextVectorization(*layer_shape, **layer_args))
                elif layer_type == 'Normalization':
                    model.add(keras.layers.Normalization(*layer_shape, **layer_args))
                elif layer_type == 'Discretization':
                    model.add(keras.layers.Discretization(*layer_shape, **layer_args))
                elif layer_type == 'CategoryEncoding':
                    model.add(keras.layers.CategoryEncoding(*layer_shape, **layer_args))
                elif layer_type == 'Hashing':
                    model.add(keras.layers.Hashing(*layer_shape, **layer_args))
                elif layer_type == 'StringLookup':
                    model.add(keras.layers.StringLookup(*layer_shape, **layer_args))
                elif layer_type == 'IntegerLookup':
                    model.add(keras.layers.IntegerLookup(*layer_shape, **layer_args))
                elif layer_type == 'Resizing':
                    model.add(keras.layers.Resizing(*layer_shape, **layer_args))
                elif layer_type == 'Rescaling':
                    model.add(keras.layers.Rescaling(*layer_shape, **layer_args))
                elif layer_type == 'CenterCrop':
                    model.add(keras.layers.CenterCrop(*layer_shape, **layer_args))
                elif layer_type == 'RandomCrop':
                    model.add(keras.layers.RandomCrop(*layer_shape, **layer_args))
                elif layer_type == 'RandomFlip':
                    model.add(keras.layers.RandomFlip(*layer_shape, **layer_args))
                elif layer_type == 'RandomTranslation':
                    model.add(keras.layers.RandomTranslation(*layer_shape, **layer_args))
                elif layer_type == 'RandomRotation':
                    model.add(keras.layers.RandomRotation(*layer_shape, **layer_args))
                elif layer_type == 'RandomZoom':
                    model.add(keras.layers.RandomZoom(*layer_shape, **layer_args))
                elif layer_type == 'RandomHeight':
                    model.add(keras.layers.RandomHeight(*layer_shape, **layer_args))
                elif layer_type == 'RandomWidth':
                    model.add(keras.layers.RandomWidth(*layer_shape, **layer_args))
                elif layer_type == 'RandomContrast':
                    model.add(keras.layers.RandomContrast(*layer_shape, **layer_args))
                elif layer_type == 'RandomBrightness':
                    model.add(keras.layers.RandomBrightness(*layer_shape, **layer_args))
                # Normalization layers:
                elif layer_type == 'BatchNormalization':
                    model.add(keras.layers.BatchNormalization(*layer_shape, **layer_args))
                elif layer_type == 'LayerNormalization':
                    model.add(keras.layers.LayerNormalization(*layer_shape, **layer_args))
                elif layer_type == 'UnitNormalization':
                    model.add(keras.layers.UnitNormalization(*layer_shape, **layer_args))
                # Regularization layers:
                elif layer_type == 'Dropout':
                    model.add(keras.layers.Dropout(*layer_shape, **layer_args))
                elif layer_type == 'SpatialDropout1D':
                    model.add(keras.layers.SpatialDropout1D(*layer_shape, **layer_args))
                elif layer_type == 'SpatialDropout2D':
                    model.add(keras.layers.SpatialDropout2D(*layer_shape, **layer_args))
                elif layer_type == 'SpatialDropout3D':
                    model.add(keras.layers.SpatialDropout3D(*layer_shape, **layer_args))
                elif layer_type == 'GaussianDropout':
                    model.add(keras.layers.GaussianDropout(*layer_shape, **layer_args))
                elif layer_type == 'GaussianNoise':
                    model.add(keras.layers.GaussianNoise(*layer_shape, **layer_args))
                elif layer_type == 'ActivityRegularization':
                    model.add(keras.layers.ActivityRegularization(*layer_shape, **layer_args))
                elif layer_type == 'AlphaDropout':
                    model.add(keras.layers.AlphaDropout(*layer_shape, **layer_args))
                # Attention layers:
                elif layer_type == 'MultiHeadAttention':
                    model.add(keras.layers.MultiHeadAttention(*layer_shape, **layer_args))
                elif layer_type == 'Attention':
                    model.add(keras.layers.Attention(*layer_shape, **layer_args))
                elif layer_type == 'AdditiveAttention':
                    model.add(keras.layers.AdditiveAttention(*layer_shape, **layer_args))
                # Reshaping layers:
                elif layer_type == 'Reshape':
                    model.add(keras.layers.Reshape(*layer_shape, **layer_args))
                elif layer_type == 'RepeatVector':
                    model.add(keras.layers.RepeatVector(*layer_shape, **layer_args))
                elif layer_type == 'Permute':
                    model.add(keras.layers.Permute(*layer_shape, **layer_args))
                elif layer_type == 'Cropping1D':
                    model.add(keras.layers.Cropping1D(*layer_shape, **layer_args))
                elif layer_type == 'Cropping2D':
                    model.add(keras.layers.Cropping2D(*layer_shape, **layer_args))
                elif layer_type == 'Cropping3D':
                    model.add(keras.layers.Cropping3D(*layer_shape, **layer_args))
                elif layer_type == 'UpSampling1D':
                    model.add(keras.layers.UpSampling1D(*layer_shape, **layer_args))
                elif layer_type == 'UpSampling2D':
                    model.add(keras.layers.UpSampling2D(*layer_shape, **layer_args))
                elif layer_type == 'UpSampling3D':
                    model.add(keras.layers.UpSampling3D(*layer_shape, **layer_args))
                elif layer_type == 'ZeroPadding1D':
                    model.add(keras.layers.ZeroPadding1D(*layer_shape, **layer_args))
                elif layer_type == 'ZeroPadding2D':
                    model.add(keras.layers.ZeroPadding2D(*layer_shape, **layer_args))
                elif layer_type == 'ZeroPadding3D':
                    model.add(keras.layers.ZeroPadding3D(*layer_shape, **layer_args))
                # Merging layers:
                elif layer_type == 'Concatenate':
                    model.add(keras.layers.Concatenate(*layer_shape, **layer_args))
                elif layer_type == 'Average':
                    model.add(keras.layers.Average(*layer_shape, **layer_args))
                elif layer_type == 'Maximum':
                    model.add(keras.layers.Maximum(*layer_shape, **layer_args))
                elif layer_type == 'Minimum':
                    model.add(keras.layers.Minimum(*layer_shape, **layer_args))
                elif layer_type == 'Add':
                    model.add(keras.layers.Add(*layer_shape, **layer_args))
                elif layer_type == 'Subtract':
                    model.add(keras.layers.Subtract(*layer_shape, **layer_args))
                elif layer_type == 'Multiply':
                    model.add(keras.layers.Multiply(*layer_shape, **layer_args))
                elif layer_type == 'Dot':
                    model.add(keras.layers.Dot(*layer_shape, **layer_args))
                # Loclally-connected layers:
                elif layer_type == 'LocallyConnected1D':
                    model.add(keras.layers.LocallyConnected1D(*layer_shape, **layer_args))
                elif layer_type == 'LocallyConnected2D':
                    model.add(keras.layers.LocallyConnected2D(*layer_shape, **layer_args))
                # Activation layers:
                elif layer_type == 'ReLU':
                    model.add(keras.layers.ReLU(*layer_shape, **layer_args))
                elif layer_type == 'Softmax':
                    model.add(keras.layers.Softmax(*layer_shape, **layer_args))
                elif layer_type == 'LeakyReLU':
                    model.add(keras.layers.LeakyReLU(*layer_shape, **layer_args))
                elif layer_type == 'PReLU':
                    model.add(keras.layers.PReLU(*layer_shape, **layer_args))
                elif layer_type == 'ELU':
                    model.add(keras.layers.ELU(*layer_shape, **layer_args))
                elif layer_type == 'ThresholdedReLU':
                    model.add(keras.layers.ThresholdedReLU(*layer_shape, **layer_args))
                elif layer_type != '':
                    importmodel = keras.models.load_model(f'{MODEL_LOCATION}/{layer_type}.h5')
                    importmodel._name = layer_type
                    model.add(importmodel)

            model.add(keras.layers.Dense(compiler.tput, activation="softmax", name='output'))

            _compile = compiler.compiler
            model.compile(**_compile)
            self.model = model

        log = Logger()
        log.start()
        model.summary()
        __msg = log.messages
        _msg = ''
        for msg in __msg:
            _msg = f'{_msg}{msg}'
        self.summary = _msg
        log.stop()

    def model_print(self):
        plot_model(self.model, to_file=f'{DRAW_MODEL_PATH}/compiled-model.gv.png', show_shapes=True,
                   show_layer_names=True)

    def fit(self, db, epoch=1, batch=1):
        xtrain = convert_to_tensor(db.dataset.xtrain)
        ytrain = convert_to_tensor(db.dataset.ytrain)
        xval = convert_to_tensor(db.dataset.xval)
        yval = convert_to_tensor(db.dataset.yval)
        history = self.model.fit(xtrain, ytrain, batch_size=batch, epochs=epoch, validation_data=(xval, yval))
        self.history = history.history
        return self.history

    @staticmethod
    def _model_scope(_devices):
        devices = []
        for _dev, role in _devices.items():
            if role == 1:
                devices.append(_dev)
        if len(devices) > 0:
            strategy = distribute.MirroredStrategy(devices)
            scope = strategy.scope()
        else:
            raise ValueError('No seelected training devices...')
        return scope


def fitmodel(model: Model, db, queue, epoch=1):
    model.model = keras.models.load_model(MODEL_TEMP)
    msg = ''
    while msg != 'MASTER:STOP':
        hist = model.fit(db, epoch, 2 ** random.randint(0, 6))
        if not queue.empty():
            msg = queue.get(False)
        queue.put(hist)
        model.model.save(MODEL_TEMP)
    return
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
