# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import matplotlib.pyplot as plt
import numpy as np

from tkinter import Tk, LabelFrame, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from ._utils import HoverButton, ColorStyles
from basenet import BaseNetDatabase, BaseNetModel


# -----------------------------------------------------------
from .__special__ import __gui_ico_path__, __models_location__, __database_location__
ICO_LOCATION = __gui_ico_path__
MODEL_LOCATION = __models_location__
DATABASE_LOCATION = __database_location__
# -----------------------------------------------------------


def guiviz():
    root_node = Tk()
    MainWindow(root_node)
    root_node.iconbitmap(ICO_LOCATION)
    root_node.configure()
    root_node.mainloop()


# -----------------------------------------------------------
class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Database Visualization")
        self.master.geometry('415x535')
        self.master.minsize(415, 535)
        self.master.maxsize(415, 535)
        self.master.configure(bg='black')
        self.colors = ColorStyles
        self.db = None

        self.canvas = None
        self.toolbar = None
        self.guess = None
        self.x = None
        self.y = None
        self.model = None

        self.canvas_lf = LabelFrame(self.master, width=400, height=430, bg='black')
        self.canvas_lf.place(x=5, y=5)
        # -------------------------------------------------------------------------------------------------------------
        #                       Buttons
        # -------------------------------------------------------------------------------------------------------------
        # Buttons:
        self.inference_button = HoverButton(self.master,
                                            text='INFERENCE WITH MODEL',
                                            command=self.infere,
                                            font='Bahnschrift 14 bold',
                                            width=36, bg=self.colors.blue)
        self.inference_button.place(x=5, y=450)

        self.newmat_button = HoverButton(self.master,
                                         text='NEW MATRIX',
                                         command=self.get_new_mat,
                                         font='Bahnschrift 14 bold',
                                         width=17, bg=self.colors.yellow)
        self.newmat_button.place(x=5, y=490)

        self.new_db_button = HoverButton(self.master,
                                         text='NEW DATABASE',
                                         command=self.new_db,
                                         font='Bahnschrift 14 bold',
                                         width=18, bg=self.colors.yellow)
        self.new_db_button.place(x=202, y=490)
        self.new_db()

    def render_mat(self, mat: np.array, ref: np.array):
        # Render RGB image.
        mesh = np.zeros((*mat.shape[:2], 3), dtype=np.uint8)
        out = np.zeros((1, *ref.shape, 3), dtype=np.uint8)
        for nr, row in enumerate(mat):
            for nc, col in enumerate(row):
                for ne, edim, in enumerate(col):
                    if ne < 3:
                        mesh[nr, nc, ne] = edim
        for ne, element in enumerate(ref):
            out[0, ne, 1] = element * 255
            if self.guess is not None:
                if self.guess.shape == ref.shape:
                    out[0, ne, 2] = self.guess[ne] * 255
                    out[0, ne, 0] = self.guess[ne] * 255
        return mesh, out

    def print_canvas(self, _mat, ref: np.array, title=''):
        # Get image.
        mat = np.concatenate((_mat, ref), axis=0)
        myfig = plt.figure(figsize=(5, 5), dpi=80)
        plt.imshow(mat, interpolation='nearest', aspect='auto')
        plt.axhline(len(mat) - 1.5, color='white')
        if title:
            plt.title(f'{title}')
        axis_x = range(len(mat) - 1)
        axis_y = range(len(mat))
        plt.xticks(axis_x, rotation='vertical')
        plt.yticks(axis_y, [*tuple(axis_x), 'Ref.'])
        # Draw canvas.
        if self.canvas is not None:
            self.canvas.get_tk_widget().pack_forget()
            self.toolbar.destroy()
        self.canvas = FigureCanvasTkAgg(myfig, master=self.canvas_lf)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_lf)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack()
        plt.close()

    def infere(self):
        if self.model is None:
            model_path = filedialog.askopenfilename(filetypes=[('Keras Model', '*.h5')], initialdir=MODEL_LOCATION)
            if model_path:
                self.model = BaseNetModel.load(model_path)
        self.guess = np.array(self.model.predict([self.x])[0])
        _x, _y = self.render_mat(self.x, self.y)
        self.print_canvas(_x, _y)

    def get_new_mat(self):
        __types = ['train', 'val', 'test']
        _type = __types[np.random.randint(0, 2)]
        no = np.random.randint(0, len(getattr(self.db, f'x{_type}')))
        x = getattr(self.db, f'x{_type}')[no] * 255
        y = getattr(self.db, f'y{_type}')[no]

        self.x = x
        self.y = y
        self.guess = np.zeros(y.shape)
        _x, _y = self.render_mat(x, y)
        self.print_canvas(_x, _y, title=f'{_type.capitalize()} matrix, number {no}')

    def new_db(self):
        new_path = filedialog.askopenfilename(filetypes=[('Database files', '*.db')], initialdir=DATABASE_LOCATION)
        if new_path:
            self.db = BaseNetDatabase.load(new_path)
            self.get_new_mat()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
