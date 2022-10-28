# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
from ._gui import gui as _gui
from ._dbgui import dbgui as _dbgui
from ._main_gui import main_gui as gui
from ._gui_viz import guiviz as _guiviz

from .database_structures import Generator
from .database_structures import HtGenerator
from .database_structures import WkGenerator

from ._report_utils import Report

from .__special__ import __cnet_user_path__, __reports_path__, \
    __models_lists_path__, __models_location__, __database_location__
import os


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


if not os.path.exists(__cnet_user_path__):
    os.mkdir(__cnet_user_path__)
if not os.path.exists(__reports_path__):
    os.mkdir(__reports_path__)
if not os.path.exists(__models_lists_path__):
    os.mkdir(__models_lists_path__)
if not os.path.exists(__models_location__):
    os.mkdir(__models_location__)
if not os.path.exists(__database_location__):
    os.mkdir(__database_location__)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
