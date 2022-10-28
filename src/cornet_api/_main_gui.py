# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from tkinter import Tk, Label, PhotoImage, BOTH
from ._utils import HoverButton, ColorStyles
from ._dbgui import dbgui
from ._gui import gui
from ._gui_viz import guiviz
from ._angui import angui
# -----------------------------------------------------------
from .__special__ import __gui_ico_path__, __logo_path__, __version__
ICO_LOCATION = __gui_ico_path__
LOGO_LOCATION = __logo_path__
VERSION = __version__
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
        self.master.geometry('300x440')
        self.master.minsize(300, 440)
        self.master.maxsize(300, 440)
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

        self.comp_button = HoverButton(self.master,
                                       text='ANALYZE MODEL',
                                       command=self.start_comp,
                                       font='Bahnschrift 14 bold',
                                       fg='black',
                                       width=25, bg=self.colors.blue)
        self.comp_button.place(x=7, y=390)

    @staticmethod
    def start_gui():
        gui()

    @staticmethod
    def start_db():
        dbgui()

    @staticmethod
    def start_viz():
        guiviz()

    @staticmethod
    def start_comp():
        angui()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
