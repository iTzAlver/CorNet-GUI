# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import pickle
import numpy as np
from tensorflow import keras, convert_to_tensor, distribute
from keras.utils.vis_utils import plot_model
from ._typeoflayers import KERAS_LISTOF_TYPEOFLAYERS
from ._logger import Logger
# -----------------------------------------------------------
import os
import json
with open('../config/config.json', 'r') as file:
    cfg = json.load(file)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['tensorflow']['devices_listing']


# -----------------------------------------------------------
class Model:
    def __init__(self, compiler, model=None):
        """
        The Class model implements an API that makes use of keras and tensorflow to build Deep Learning Models.
        :param compiler: Compiler object to build the model.
        :param model: If a keras.model is already compiled, you can import it in the model parameter, so the compiler
        will not be used.
        """
        self.model = None
        self.compiler = compiler
        self.devices: dict = compiler.devices
        self.summary: str = 'Uncompiled model.'
        if model is None:
            self.compile()
        else:
            self.model = model
        self._logtracker()
        self.history: list = []
        self.is_trained: bool = False
        self._scope = None

    def compile(self):
        compiler = self.compiler
        self._scope = self._model_scope(self.devices)

        with self._scope:
            # Add the input of the model.
            _inp = keras.Input(shape=compiler.io_shape[0])
            _inp._name = 'compiled-model-keras'
            _lastlay = _inp

            last_master = None
            pipeline_opened = False
            towers = []

            for layer_type, layer_shape, layer_args in zip(compiler.layers, compiler.shapes, compiler.args):
                # Core layers:
                if layer_type == 'open_pipeline':
                    if not pipeline_opened:
                        last_master = _lastlay
                    else:
                        towers.append(_lastlay)
                        _lastlay = last_master
                    pipeline_opened = True
                elif layer_type == 'close_pipeline':
                    pipeline_opened = False
                    towers.append(_lastlay)
                    _lastlay = keras.layers.concatenate(towers, **layer_args)
                    towers = []
                else:
                    if layer_type in KERAS_LISTOF_TYPEOFLAYERS:
                        this_lay = getattr(keras.layers, layer_type)(*layer_shape, **layer_args)
                    else:
                        importmodel = keras.models.load_model(f'{layer_type}.h5')
                        importmodel._name = layer_type
                        this_lay = importmodel

                    _lastlay = this_lay(_lastlay)

            # Add the output of the model.
            out = keras.layers.Dense(compiler.io_shape[1], activation="softmax", name='output')(_lastlay)

            _compile = compiler.compiler
            model = keras.Model(_inp, out)
            model.compile(**_compile)
            self.model = model

    def model_print(self, print_path):
        plot_model(self.model, to_file=f'{print_path}/compiled-model.gv.png', show_shapes=True)

    def fit(self, db, epoch=1):
        self.is_trained = True
        xtrain = convert_to_tensor(np.array(db.dataset.xtrain).astype("float32") / 255)
        ytrain = convert_to_tensor(db.dataset.ytrain)
        xval = convert_to_tensor(np.array(db.dataset.xval).astype("float32") / 255)
        yval = convert_to_tensor(db.dataset.yval)
        history = self.model.fit(xtrain, ytrain, batch_size=db.batch_size, epochs=epoch, validation_data=(xval, yval))
        self.history.append(history.history)
        return history.history

    def save(self, model_path, compiler_path=''):
        self.model.save(model_path)
        if not compiler_path:
            _compiler_path = model_path.replace('.h5', '.cpl')
        elif '.cpl' in compiler_path:
            _compiler_path = compiler_path
        else:
            _compiler_path = f'{compiler_path}.cpl'
        with open(_compiler_path, 'wb') as file:
            pickle.dump(self.compiler, file)

    @staticmethod
    def load(model_path, compiler_path=''):
        model = keras.models.load_model(model_path)
        if not compiler_path:
            _compiler_path = model_path.replace('.h5', '.cpl')
        elif '.cpl' in compiler_path:
            _compiler_path = compiler_path
        else:
            _compiler_path = f'{compiler_path}.cpl'
        with open(_compiler_path, 'rb') as file:
            compiler = pickle.load(file)
        return Model(compiler, model=model)

    @staticmethod
    def _model_scope(_devices):
        # Cretes a scope from the current devices.
        devices = []
        for _dev, role in _devices.items():
            if role == 1:
                devices.append(_dev)
        if len(devices) > 0:
            strategy = distribute.MirroredStrategy(devices=devices)
            scope = strategy.scope()
        else:
            raise ValueError('There are no training devices...')
        return scope

    def _logtracker(self):
        # Extracts the summary of the current model.
        log = Logger()
        log.start()
        self.model.summary()
        __msg = log.messages
        _msg = ''
        for msg in __msg:
            _msg = f'{_msg}{msg}'
        self.summary = _msg
        log.stop()

    @staticmethod
    def fitmodel(_model, db, queue, bypass='', epoch=1):
        if bypass:
            model = Model.load(bypass)
        else:
            model = _model
        msg = ''
        while msg != 'MASTER:STOP':
            hist = model.fit(db, epoch)
            if not queue.empty():
                msg = queue.get()
            queue.put(hist)
        if bypass:
            model.save(bypass)
        return

    def __repr__(self):
        return f'Model object with the following parameters:\nCompiler: {self.compiler}\nSummary: {self.summary}'

    def __bool__(self):
        return self.is_trained

    def __sizeof__(self):
        return len(self.compiler.layers)

    def __eq__(self, other):
        return self.compiler.layers == other.compiler.layers
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
