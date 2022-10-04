# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import sys
from tensorflow import keras


class Logger:
    stdout = sys.stdout
    messages = []

    def start(self):
        sys.stdout = self
        self.messages = []

    def stop(self):
        sys.stdout = self.stdout

    def write(self, text):
        self.messages.append(text)

    def flush(self):
        sys.stdout.flush()
        self.stdout.flush()
        self.flush()


def layers_help(layer_type: str) -> str:
    log = Logger()
    log.start()
    if layer_type == 'Dense':
        help(keras.layers.Dense)
    elif layer_type == 'Flatten':
        help(keras.layers.Flatten)
    elif layer_type == 'Activation':
        help(keras.layers.Activation)
    elif layer_type == 'Embedding':
        help(keras.layers.Embedding)
    elif layer_type == 'Masking':
        help(keras.layers.Masking)
    elif layer_type == 'Lambda':
        help(keras.layers.Lambda)
    # Convolutional layers:
    elif layer_type == 'Conv1D':
        help(keras.layers.Conv1D)
    elif layer_type == 'Conv2D':
        help(keras.layers.Conv2D)
    elif layer_type == 'Conv3D':
        help(keras.layers.Conv3D)
    elif layer_type == 'SeparableConv1D':
        help(keras.layers.SeparableConv1D)
    elif layer_type == 'SeparableConv2D':
        help(keras.layers.SeparableConv2D)
    elif layer_type == 'DepthwiseConv2D':
        help(keras.layers.DepthwiseConv2D)
    elif layer_type == 'Conv1DTranspose':
        help(keras.layers.Conv1DTranspose)
    elif layer_type == 'Conv2DTranspose':
        help(keras.layers.Conv2DTranspose)
    elif layer_type == 'Conv3DTranspose':
        help(keras.layers.Conv3DTranspose)
    # Pooling layers:
    elif layer_type == 'MaxPooling1D':
        help(keras.layers.MaxPooling1D)
    elif layer_type == 'MaxPooling2D':
        help(keras.layers.MaxPooling2D)
    elif layer_type == 'MaxPooling3D':
        help(keras.layers.MaxPooling3D)
    elif layer_type == 'AveragePooling1D':
        help(keras.layers.AveragePooling1D)
    elif layer_type == 'AveragePooling2D':
        help(keras.layers.AveragePooling2D)
    elif layer_type == 'AveragePooling3D':
        help(keras.layers.AveragePooling3D)
    elif layer_type == 'GlobalMaxPooling1D':
        help(keras.layers.GlobalMaxPooling1D)
    elif layer_type == 'GlobalMaxPooling2D':
        help(keras.layers.GlobalMaxPooling2D)
    elif layer_type == 'GlobalMaxPooling3D':
        help(keras.layers.GlobalMaxPooling3D)
    elif layer_type == 'GlobalAveragePooling1D':
        help(keras.layers.GlobalAveragePooling1D)
    elif layer_type == 'GlobalAveragePooling2D':
        help(keras.layers.GlobalAveragePooling2D)
    elif layer_type == 'GlobalAveragePooling3D':
        help(keras.layers.GlobalAveragePooling3D)
    # Recursant layers.
    elif layer_type == 'LSTM':
        help(keras.layers.LSTM)
    elif layer_type == 'GRU':
        help(keras.layers.GRU)
    elif layer_type == 'SimpleRNN':
        help(keras.layers.SimpleRNN)
    elif layer_type == 'RNN':
        help(keras.layers.RNN)
    elif layer_type == 'TimeDistributed':
        help(keras.layers.TimeDistributed)
    elif layer_type == 'Bidirectional':
        help(keras.layers.Bidirectional)
    elif layer_type == 'ConvLSTM1D':
        help(keras.layers.ConvLSTM1D)
    elif layer_type == 'ConvLSTM2D':
        help(keras.layers.ConvLSTM2D)
    elif layer_type == 'ConvLSTM3D':
        help(keras.layers.ConvLSTM3D)
    # Preprocessing layers:
    elif layer_type == 'TextVectorization':
        help(keras.layers.TextVectorization)
    elif layer_type == 'Normalization':
        help(keras.layers.Normalization)
    elif layer_type == 'Discretization':
        help(keras.layers.Discretization)
    elif layer_type == 'CategoryEncoding':
        help(keras.layers.CategoryEncoding)
    elif layer_type == 'Hashing':
        help(keras.layers.Hashing)
    elif layer_type == 'StringLookup':
        help(keras.layers.StringLookup)
    elif layer_type == 'IntegerLookup':
        help(keras.layers.IntegerLookup)
    elif layer_type == 'Resizing':
        help(keras.layers.Resizing)
    elif layer_type == 'Rescaling':
        help(keras.layers.Rescaling)
    elif layer_type == 'CenterCrop':
        help(keras.layers.CenterCrop)
    elif layer_type == 'RandomCrop':
        help(keras.layers.RandomCrop)
    elif layer_type == 'RandomFlip':
        help(keras.layers.RandomFlip)
    elif layer_type == 'RandomTranslation':
        help(keras.layers.RandomTranslation)
    elif layer_type == 'RandomRotation':
        help(keras.layers.RandomRotation)
    elif layer_type == 'RandomZoom':
        help(keras.layers.RandomZoom)
    elif layer_type == 'RandomHeight':
        help(keras.layers.RandomHeight)
    elif layer_type == 'RandomWidth':
        help(keras.layers.RandomWidth)
    elif layer_type == 'RandomContrast':
        help(keras.layers.RandomContrast)
    elif layer_type == 'RandomBrightness':
        help(keras.layers.RandomBrightness)
    # Normalization layers:
    elif layer_type == 'BatchNormalization':
        help(keras.layers.BatchNormalization)
    elif layer_type == 'LayerNormalization':
        help(keras.layers.LayerNormalization)
    elif layer_type == 'UnitNormalization':
        help(keras.layers.UnitNormalization)
    # Regularization layers:
    elif layer_type == 'Dropout':
        help(keras.layers.Dropout)
    elif layer_type == 'SpatialDropout1D':
        help(keras.layers.SpatialDropout1D)
    elif layer_type == 'SpatialDropout2D':
        help(keras.layers.SpatialDropout2D)
    elif layer_type == 'SpatialDropout3D':
        help(keras.layers.SpatialDropout3D)
    elif layer_type == 'GaussianDropout':
        help(keras.layers.GaussianDropout)
    elif layer_type == 'GaussianNoise':
        help(keras.layers.GaussianNoise)
    elif layer_type == 'ActivityRegularization':
        help(keras.layers.ActivityRegularization)
    elif layer_type == 'AlphaDropout':
        help(keras.layers.AlphaDropout)
    # Attention layers:
    elif layer_type == 'MultiHeadAttention':
        help(keras.layers.MultiHeadAttention)
    elif layer_type == 'Attention':
        help(keras.layers.Attention)
    elif layer_type == 'AdditiveAttention':
        help(keras.layers.AdditiveAttention)
    # Reshaping layers:
    elif layer_type == 'Reshape':
        help(keras.layers.Reshape)
    elif layer_type == 'RepeatVector':
        help(keras.layers.RepeatVector)
    elif layer_type == 'Permute':
        help(keras.layers.Permute)
    elif layer_type == 'Cropping1D':
        help(keras.layers.Cropping1D)
    elif layer_type == 'Cropping2D':
        help(keras.layers.Cropping2D)
    elif layer_type == 'Cropping3D':
        help(keras.layers.Cropping3D)
    elif layer_type == 'UpSampling1D':
        help(keras.layers.UpSampling1D)
    elif layer_type == 'UpSampling2D':
        help(keras.layers.UpSampling2D)
    elif layer_type == 'UpSampling3D':
        help(keras.layers.UpSampling3D)
    elif layer_type == 'ZeroPadding1D':
        help(keras.layers.ZeroPadding1D)
    elif layer_type == 'ZeroPadding2D':
        help(keras.layers.ZeroPadding2D)
    elif layer_type == 'ZeroPadding3D':
        help(keras.layers.ZeroPadding3D)
    # Merging layers:
    elif layer_type == 'Concatenate':
        help(keras.layers.Concatenate)
    elif layer_type == 'Average':
        help(keras.layers.Average)
    elif layer_type == 'Maximum':
        help(keras.layers.Maximum)
    elif layer_type == 'Minimum':
        help(keras.layers.Minimum)
    elif layer_type == 'Add':
        help(keras.layers.Add)
    elif layer_type == 'Subtract':
        help(keras.layers.Subtract)
    elif layer_type == 'Multiply':
        help(keras.layers.Multiply)
    elif layer_type == 'Dot':
        help(keras.layers.Dot)
    # Loclally-connected layers:
    elif layer_type == 'LocallyConnected1D':
        help(keras.layers.LocallyConnected1D)
    elif layer_type == 'LocallyConnected2D':
        help(keras.layers.LocallyConnected2D)
    # Activation layers:
    elif layer_type == 'ReLU':
        help(keras.layers.ReLU)
    elif layer_type == 'Softmax':
        help(keras.layers.Softmax)
    elif layer_type == 'LeakyReLU':
        help(keras.layers.LeakyReLU)
    elif layer_type == 'PReLU':
        help(keras.layers.PReLU)
    elif layer_type == 'ELU':
        help(keras.layers.ELU)
    elif layer_type == 'ThresholdedReLU':
        help(keras.layers.ThresholdedReLU)
    __msg = log.messages
    _msg = ''
    for msg in __msg:
        _msg = f'{_msg}{msg}'
    summary = _msg
    log.stop()
    return summary

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
