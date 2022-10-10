# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from tensorflow import keras, distribute
import numpy as np
import time



# -----------------------------------------------------------
def main() -> None:
    strategy = distribute.MirroredStrategy(devices=['/device:GPU:0'])
    scope = strategy.scope()
    with scope:
        io_shape = ((28, 28, 1), 10)
        model = keras.Sequential()
        model.add(keras.Input(shape=io_shape[0]))
        model.add(keras.layers.Conv2D(64, 3))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(180))
        model.add(keras.layers.Dense(io_shape[1]))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    (xtrain, ytrain), (_, _) = keras.datasets.mnist.load_data()
    xtrain = xtrain.astype("float32") / 255
    xtrain = np.expand_dims(xtrain, -1)
    ytrain = keras.utils.to_categorical(ytrain, 10)
    tim0 = time.perf_counter()
    model.fit(xtrain, ytrain, batch_size=128)
    tim1 = time.perf_counter()
    print(tim1-tim0)  # 47.72
    return


if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
