# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from tkinter import Tk, Label, PhotoImage, BOTH
from .utils import HoverButton, ColorStyles
from ._dbgui import dbgui
from ._gui import gui
from ._gui_viz import guiviz
# -----------------------------------------------------------
import json
from .__path_to_config__ import PATH_TO_CONFIG
try:
    with open(PATH_TO_CONFIG, 'r') as _file:
        cfg = json.load(_file)
        ICO_LOCATION = cfg["path"]["ICO_LOCATION"]
        LOGO_LOCATION = cfg["path"]["LOGO_LOCATION"]
        VERSION = cfg["version"]
except Exception as _ex:
    print('Traceback: _gui.py. Path to config corrupted, try to run the gui from the proper executable.')
    raise _ex


# -----------------------------------------------------------
def main_gui() -> None:
    root_node = Tk()
    MainWindow(root_node)
    root_node.iconbitmap(ICO_LOCATION)
    root_node.configure()
    root_node.mainloop()


class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Cornet API v0")
        self.master.geometry('300x400')
        self.master.minsize(300, 400)
        self.master.maxsize(300, 400)
        self.master.configure(bg='black')
        self.colors = ColorStyles

        self.img = PhotoImage(file=LOGO_LOCATION)
        self.img = self.img.subsample(2)
        self.logo = Label(self.master, image=self.img, bg='black')
        self.logo.image = self.img
        self.logo.pack(fill=BOTH, expand=1)
        self.logo.place(x=-20, y=0)

        self.labelcornet = Label(self.master, text=f'CorNet API v{VERSION}', fg='white', bg='black',
                                 font='Fixedsys 21 bold')
        self.labelcornet.place(x=3, y=180)

        self.gui_button_ = HoverButton(self.master,
                                       text='MODEL BUILDING AND TRAINING',
                                       command=self.start_gui,
                                       font='Bahnschrift 14 bold',
                                       fg='black',
                                       width=25, bg=self.colors.yellow)
        self.gui_button_.place(x=7, y=240)

        self.ddbb_button = HoverButton(self.master,
                                       text='DATABASE BUILDER',
                                       command=self.start_db,
                                       font='Bahnschrift 14 bold',
                                       fg='black',
                                       width=25, bg=self.colors.red)
        self.ddbb_button.place(x=7, y=290)

        self.dviz_button = HoverButton(self.master,
                                       text='DATABASE VISUALIZATION',
                                       command=self.start_viz,
                                       font='Bahnschrift 14 bold',
                                       fg='black',
                                       width=25, bg=self.colors.orange)
        self.dviz_button.place(x=7, y=340)

    @staticmethod
    def start_gui():
        gui()

    @staticmethod
    def start_db():
        dbgui()

    @staticmethod
    def start_viz():
        guiviz()


if __name__ == '__main__':
    main_gui()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #