# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
import os
__version__ = '0.3.3'
__specials_path__ = os.path.abspath(f'{__file__.replace(f"__special__.py", "")}')
__include_path__ = f'{__specials_path__}/include/'

__latex_path__ = f'{__include_path__}latex/'

__gui_ico_path__ = f'{__include_path__}multimedia/uah.ico'
__logo_path__ = f'{__include_path__}multimedia/cornet.png'
__render_model_path__ = f'{__include_path__}multimedia/render/'

__logfile_path__ = f'{__include_path__}temp/logfile.txt'
__db_logfile_path__ = f'{__include_path__}temp/logfiledb.txt'
__language_models_path__ = f'{__include_path__}temp/lms.txt'

__latex_compile_command__ = "pdflatex main.tex"

__home_user_path__ = os.path.expanduser('~')
__cnet_user_path__ = f'{__home_user_path__}/CorNet/'
__reports_path__ = f'{__cnet_user_path__}/reports'
__models_lists_path__ = f'{__cnet_user_path__}/model_lists'
__models_location__ = f'{__cnet_user_path__}/models'
__database_location__ = f'{__cnet_user_path__}/db'

__wikipedia_random_path__ = 'http://es.wikipedia.org/wiki/Special:Random'
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
