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
    
    
class Report:
    def __init__(self, image_path, latex_path, model, db, history):
        self.author = 'Palomo Alonso, Alberto'
        self._latex_path = latex_path
        self.noreport = self._getnumber()
        self.model_name = model.model._name.replace('_', ' ')
        self.dbname = db.name.replace('_', ' ')
        self.dbtype = db.type.replace('_', ' ')
        self.dbsize = db.size
        self.dbdist = db.distribution
        self.model_input = model.compiler.io_shape[0]
        self.model_output = model.compiler.io_shape[1]
        self.model_loss_func = model.compiler.compiler['loss'].replace('_', ' ')
        self.model_optimizer = model.compiler.compiler['optimizer'].replace('_', ' ')
        ndevs = 0
        for _, role in model.devices.items():
            if role == 1:
                ndevs += 1
        self.training_devs = ndevs
        if history:
            self.max_loss = round(max(history)[0], 4)
            self.min_loss = round(min(history)[0], 4)
        else:
            self.max_loss = '???'
            self.min_loss = '???'
        self.model_in_tab = self._tabularize(model.compiler.layers, model.compiler.shapes, model.compiler.args)

        self._movefiles(latex_path, image_path, self._print_report(), model.summary, history)
        self._compile()

    def _compile(self):
        try:
            os.remove('../temp/latex/main.pdf')
        except Exception as ex:
            print(ex)
        os.system(f'cd ../temp/latex & pdflatex main.tex')
        shutil.copyfile('../temp/latex/main.pdf', f'../doc/reports/{self.noreport}.pdf')

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
    def _tabularize(layers, shapes, args):
        _text = ''
        for layer, shape, arg in zip(layers, shapes, args):
            if arg:
                _text = f'{_text}{layer} & {shape} & {arg} \\\\ \n'
            else:
                _text = f'{_text}{layer} & {shape} &  \\\\ \n'
        return _text

    @staticmethod
    def _movefiles(latex, model, report, summary, history):
        shutil.copyfile(model, f'{latex}/imgs/model.png')
        saveas = f'{latex}/imgs/learning.png'
        with open(f'{latex}/commands.tex', 'w', encoding='utf-8') as file:
            file.write(report)
        with open(f'{latex}/summary.txt', 'w', encoding='utf-8') as file:
            file.write(summary)
        myfig = plt.figure(figsize=(4.86, 3), dpi=80)
        plt.plot(history, 'k')
        plt.title('Learning curve')
        plt.grid()
        plt.savefig(saveas)
        plt.close(myfig)

    def _getnumber(self):
        with open(f'{self._latex_path}/number', 'r', encoding='utf-8') as file:
            current = int(file.readline().replace('\n', ''))
            number = current + 1
            number_str = str(number)
            number_str = '0' * (4 - len(number_str)) + number_str
        with open(f'{self._latex_path}/number', 'w', encoding='utf-8') as file:
            file.write(f'{number}')
        return f'CNET{number_str}'
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
