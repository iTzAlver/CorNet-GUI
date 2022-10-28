# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from tkinter import Button


class HoverButton(Button):
    def __init__(self, master, **kw):
        Button.__init__(self, master=master, **kw)
        self.defaultBackground = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.brigth_span = 3
        self.newBG = '#'

    def on_enter(self, e):
        self.newBG = '#'
        for index in range(len(self["background"])):
            if index != 0:
                hex_str = str(hex(int(self["background"][index], 16) + self.brigth_span))
                self.newBG = self.newBG + hex_str[2]
        self['background'] = self.newBG
        return e

    def on_leave(self, e):
        self.newBG = '#'
        for index in range(len(self["background"])):
            if index != 0:
                hex_str = str(hex(int(self["background"][index], 16) - self.brigth_span))
                self.newBG = self.newBG + hex_str[2]
        self['background'] = self.newBG
        return e
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
