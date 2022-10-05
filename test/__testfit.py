# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from tensorflow import keras, convert_to_tensor
from src._database_structure import load_database
# Import the required libraries
from tkinter import *
from tkinter import messagebox
import time


# -----------------------------------------------------------
def main() -> None:
    model = keras.models.load_model('../db/models/test.h5')
    db = load_database('../db/db/ht/lk.ht')
    xtrain = convert_to_tensor(db.dataset.xtrain)
    ytrain = convert_to_tensor(db.dataset.ytrain)
    xval = convert_to_tensor(db.dataset.xval)
    yval = convert_to_tensor(db.dataset.yval)
    history = model.fit(xtrain, ytrain, batch_size=1, epochs=2, validation_data=(xval, yval))
    history = history.history
    print(history)
    return


def main2():

    # Create an instance of tkinter frame or window
    win = Tk()

    # Set the geometry
    win.geometry("700x200")

    # Define the function for button
    def some_task():
        time.sleep(100)
        # Recursively call the function
        win.after(2000, some_task)

    # Keep Running the window
    win.after(2000, some_task)
    win.mainloop()


if __name__ == '__main__':
    main2()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
