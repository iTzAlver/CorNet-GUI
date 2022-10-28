# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from tkinter import Tk, LabelFrame, filedialog
from ._utils import HoverButton, ColorStyles
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from basenet import BaseNetDatabase, BaseNetModel
from basenet import window_diff
# -----------------------------------------------------------
from .__special__ import __gui_ico_path__, __models_location__, __database_location__
ICO_LOCATION = __gui_ico_path__
MODEL_LOCATION = __models_location__
DATABASE_LOCATION = __database_location__
# -----------------------------------------------------------


def angui() -> None:
    root_node = Tk()
    MainWindow(root_node)
    root_node.iconbitmap(ICO_LOCATION)
    root_node.configure()
    root_node.mainloop()


# -----------------------------------------------------------
class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Model analysis")
        self.master.geometry('415x495')
        self.master.minsize(415, 495)
        self.master.maxsize(415, 495)
        self.master.configure(bg='black')
        self.colors = ColorStyles

        self.dbs = None
        self.canvas = None
        self.toolbar = None
        self.results = None
        self.model: (BaseNetModel, None) = None
        self.names = []

        self.canvas_lf = LabelFrame(self.master, width=400, height=430, bg='black')
        self.canvas_lf.place(x=5, y=5)
        # -------------------------------------------------------------------------------------------------------------
        #                       Buttons
        # -------------------------------------------------------------------------------------------------------------
        # Buttons:
        self.newmat_button = HoverButton(self.master,
                                         text='NEW MODEL',
                                         command=self.select_model,
                                         font='Bahnschrift 14 bold',
                                         width=17, bg=self.colors.blue)
        self.newmat_button.place(x=5, y=450)

        self.new_db_button = HoverButton(self.master,
                                         text='NEW DATABASE',
                                         command=self.select_db,
                                         font='Bahnschrift 14 bold',
                                         width=18, bg=self.colors.yellow)
        self.new_db_button.place(x=202, y=450)
        self._select_model()
        self.select_db()

    def print_canvas(self, x: np.array, y: list):
        # Get image.
        colors = cm.jet(np.linspace(0, 1, len(y)))
        myfig = plt.figure(figsize=(5, 5), dpi=80)
        minimum = []
        minimum_index = []
        for nidx, dbsol in enumerate(y):
            plt.plot(x, dbsol, label=self.names[nidx], color=colors[nidx])
            minimum.append(min(dbsol))
            minimum_index.append(x[np.argmin(dbsol)])
        plt.legend()
        plt.scatter(minimum_index, minimum, color='k')
        plt.title('Model performance evaluation')
        plt.grid(b=True, color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-')
        plt.ylim(0, 1.05)
        plt.xlim(0, 1)
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

    def select_db(self):
        new_paths = filedialog.askopenfilenames(filetypes=[('Database files', '*.db')], initialdir=DATABASE_LOCATION)
        if new_paths:
            self.dbs = []
            self.names = []
            for new_path in new_paths:
                this_db = BaseNetDatabase.load(new_path)
                self.dbs.append(this_db)
                self.model.add_database(this_db)
                self.names.append(new_path.split('/')[-1].replace('.db', ''))
        self.analyze()

    def _select_model(self):
        new_path = filedialog.askopenfilename(filetypes=[('Keras model', '*.h5')], initialdir=MODEL_LOCATION)
        if new_path:
            self.model: BaseNetModel = BaseNetModel.load(new_path)

    def select_model(self):
        self._select_model()
        self.analyze()

    def analyze(self, segmentation=100):
        x_wd = np.linspace(0, 1, segmentation + 1)
        y_wd = []
        for ix, db in enumerate(self.model.breech):
            print(f'Importing database: {db.name}.')
            solution_for_th = []
            for thval in x_wd:
                solution_for_th.append(self.model.evaluate(ix, window_diff, th=thval))
                self.master.update_idletasks()
                self.master.update()
            y_wd.append(solution_for_th)
        self.print_canvas(x_wd, y_wd)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
