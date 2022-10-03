# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
from ._gui import gui
from ._model import Model
from ._compiler import Compiler


def info():
    __text = f'CORNET package:\n' \
             f'--------------------------------------------\n' \
             f'FAST USAGE:\n' \
             f'--------------------------------------------\n' \
             f'from cornet import cornetgui as cgui\n' \
             f'cgui()\n' \
             f'--------------------------------------------\n'
    print(__text)
    return __text


def about():
    __text = f'Institution:\n' \
             f'------------------------------------------------------\n' \
             f'Universidad de Alcalá.\n' \
             f'Escuela Politécnica Superior.\n' \
             f'Departamento de Teoría De la Señal y Comunicaciones.\n' \
             f'Cátedra ISDEFE.\n' \
             f'------------------------------------------------------\n' \
             f'Author: Alberto Palomo Alonso\n' \
             f'------------------------------------------------------\n'
    print(__text)
    return __text
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
