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
        self.noreport = 'CNET0001'
        self.model_name = model._name
        self.dbname = db.name
        self.dbtype = db.type
        self.dbsize = db.size
        self.dbdist = db.dist
        self.model_input = model.compiler.io_shape[0]
        self.model_output = model.compiler.io_shape[1]
        self.model_loss_func = model.compiler.compiler['loss']
        self.model_optimizer = model.compiler.compiler['optimizer']
        ndevs = 0
        for _, role in model.devices:
            if role == 1:
                ndevs += 1
        self.training_devs = ndevs
        self.max_loss = max(history)
        self.min_loss = min(history)
        self.model_in_tab = self._tabularize(model.layers, model.shapes, model.args)

        self._latex_path = latex_path
        self._movefiles(latex_path, image_path, self._print_report(), model.summary, history)
        self._compile()

    def _compile(self):
        os.system(f'pdflatex {self._latex_path}/main.tex')

    def _print_report(self):
        _text = r'\newcommand\{\authorx}' \
                f'{self.author}\n' \
                r'\newcommand{\noreport}' \
                f'{self.noreport}\n' \
                r'\newcommand{\name}' \
                f'{self.model_name}\n' \
                r'\newcommand{\dbname}' \
                f'{self.dbname}\n' \
                r'\newcommand{\dbtype}' \
                f'{self.dbtype}\n' \
                r'\newcommand{\dbsize}' \
                f'{self.dbsize}\n' \
                r'\newcommand{\dbdist}' \
                f'{self.dbdist}\n' \
                r'\newcommand{\modeli}' \
                f'{self.model_input}\n' \
                r'\newcommand{\modelo}' \
                f'{self.model_output}\n' \
                r'\newcommand{\modelloss}' \
                f'{self.model_loss_func}\n' \
                r'\newcommand{\modelopt}' \
                f'{self.model_optimizer}\n' \
                r'\newcommand{\modeldevs}' \
                f'{self.training_devs}\n' \
                r'\newcommand{\modeltab}' \
                f'{self.model_in_tab}\n' \
                r'\newcommand{\maxloss}' \
                f'{self.max_loss}\n' \
                r'\newcommand{\minloss}' \
                f'{self.min_loss}\n'
        return _text

    @staticmethod
    def _tabularize(layers, shapes, args):
        _text = ''
        for layer, shape, arg in zip(layers, shapes, args):
            _text = f'{_text}{layer} & {shape} & {args} \\'
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
        plt.plot(history)
        plt.title('Learning curve')
        plt.grid()
        plt.savefig(saveas)
        plt.close(myfig)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
