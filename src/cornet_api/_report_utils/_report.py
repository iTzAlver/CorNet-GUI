# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import os
import shutil
import matplotlib.pyplot as plt
from ..__special__ import __latex_path__, __latex_compile_command__, __reports_path__


class Report:
    def __init__(self, image_path, model, db, history, author='Palomo Alonso, Alberto', number=None):
        self.author = author
        if number:
            self.noreport = number
        else:
            self.noreport = self._getnumber()
        self.model_name = model.model.name.replace('_', ' ')
        self.dbname = db.name.replace('_', ' ')
        self.dbtype = 'BaseNetDatabase (BND) '
        self.dbsize = sum(db.size)
        self.dbdist = db.distribution
        self.model_input = model.compiler.io_shape[0]
        self.model_output = model.compiler.io_shape[1]
        self.model_loss_func = model.compiler.compile_options['loss'].replace('_', ' ')
        self.model_optimizer = model.compiler.compile_options['optimizer'].replace('_', ' ')
        ndevs = 0
        for _, role in model.compiler.devices.items():
            if role == 'Train':
                ndevs += 1
        self.training_devs = ndevs
        if history:
            self.max_loss = round(max(history['loss']), 4)
            self.min_loss = round(min(history['loss']), 4)
        else:
            self.max_loss = '???'
            self.min_loss = '???'
        self.model_in_tab = self._tabularize(model.compiler.layers)

        self._movefiles(image_path, self._print_report(), model.summary, history)
        self._compile()

    def _compile(self):
        try:
            os.remove(f'{__latex_path__}main.pdf')
        except Exception as ex:
            print(ex)
        os.system(f'cd {__latex_path__} & {__latex_compile_command__}')
        try:
            shutil.copyfile(f'{__latex_path__}main.pdf', f'{__reports_path__}/{self.noreport}.pdf')
        except Exception as ex:
            print(ex)
            print('It looks like you did not install LiveTeX correctly. Make sure that the command '
                  f'{__latex_compile_command__} is actually working on your machine. Don\'t worry if not, '
                  f'this feature is not mandatory.')

    def _print_report(self):
        _text = r'\newcommand{\authorx}{' \
                f'{self.author}' \
                '}\n'\
                r'\newcommand{\noreport}{' \
                f'{self.noreport}' \
                '}\n'\
                r'\newcommand{\name}{' \
                f'{self.model_name}' \
                '}\n'\
                r'\newcommand{\dbname}{' \
                f'{self.dbname}' \
                '}\n'\
                r'\newcommand{\dbtype}{' \
                f'{self.dbtype}' \
                '}\n'\
                r'\newcommand{\dbsize}{' \
                f'{self.dbsize}' \
                '}\n'\
                r'\newcommand{\dbdist}{' \
                f'{self.dbdist}' \
                '}\n'\
                r'\newcommand{\modeli}{' \
                f'{self.model_input}' \
                '}\n' \
                r'\newcommand{\modelo}{' \
                f'{self.model_output}' \
                '}\n'\
                r'\newcommand{\modelloss}{' \
                f'{self.model_loss_func}' \
                '}\n'\
                r'\newcommand{\modelopt}{' \
                f'{self.model_optimizer}' \
                '}\n'\
                r'\newcommand{\modeldevs}{' \
                f'{self.training_devs}' \
                '}\n'\
                r'\newcommand{\modeltab}{' \
                f'{self.model_in_tab}' \
                '}\n' \
                r'\newcommand{\maxloss}{' \
                f'{self.max_loss}' \
                '}\n' \
                r'\newcommand{\minloss}{' \
                f'{self.min_loss}' \
                '}\n'
        return _text

    @staticmethod
    def _tabularize(layers):
        _text = ''
        for _layer in layers:
            for layer, items in _layer.items():
                shape = items[0]
                arg = items[1]
                if arg:
                    _text = f'{_text}{layer} & {shape} & {arg} \\\\ \n'
                else:
                    _text = f'{_text}{layer} & {shape} &  \\\\ \n'
        return _text

    @staticmethod
    def _movefiles(model, report, summary, history):
        shutil.copyfile(model, f'{__latex_path__}/imgs/model.png')
        saveas = f'{__latex_path__}/imgs/learning.png'
        with open(f'{__latex_path__}/commands.tex', 'w', encoding='utf-8') as file:
            file.write(report)
        with open(f'{__latex_path__}/summary.txt', 'w', encoding='utf-8') as file:
            file.write(summary)
        if history:
            myfig = plt.figure(figsize=(4.86, 3), dpi=80)
            plt.plot(history['loss'], 'b', label='Training')
            plt.plot(history['val_loss'], 'r', label='Validation')
            plt.legend()
            plt.title('Learning curve')
            plt.grid(b=True, color='#666666', linestyle='-')
            plt.minorticks_on()
            plt.grid(b=True, which='minor', color='#999999', linestyle='-')
            plt.ylim(0, max(history['loss']) * 1.05)
            plt.savefig(saveas)
            plt.close(myfig)

    @staticmethod
    def _getnumber():
        with open(f'{__latex_path__}/number', 'r', encoding='utf-8') as file:
            current = int(file.readline().replace('\n', ''))
            number = current + 1
            number_str = str(number)
            number_str = '0' * (4 - len(number_str)) + number_str
        with open(f'{__latex_path__}/number', 'w', encoding='utf-8') as file:
            file.write(f'{number}')
        return f'CNET{number_str}'
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
