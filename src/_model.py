# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import sys
from tensorflow import keras
from keras.utils.vis_utils import plot_model

DRAW_MODEL_PATH = r'../multimedia/render'


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
        self._compiler = compiler
        self.devices = compiler.devices
        self._compile()

    def _compile(self):
        compiler = self._compiler
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
            # Special layers:
            elif layer_type == 'TextVectorization':
                model.add(keras.layers.TextVectorization(*layer_shape, **layer_args))
            elif layer_type == 'Normalization':
                model.add(keras.layers.Normalization(*layer_shape, **layer_args))
            elif layer_type == 'Discretization':
                model.add(keras.layers.Discretization(*layer_shape, **layer_args))
            elif layer_type == 'Dropout':
                model.add(keras.layers.Dropout(*layer_shape, **layer_args))

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

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
