# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import multiprocessing
import time
from sentence_transformers import SentenceTransformer
from tkinter import Tk, LabelFrame, Label, Entry, END, scrolledtext, filedialog, ttk
from ._utils import ColorStyles, HoverButton
from .database_structures import HtGenerator, WkGenerator
from .database_structures import Generator

# -----------------------------------------------------------
from .__special__ import __gui_ico_path__, __db_logfile_path__, __language_models_path__, __database_location__
ICO_LOCATION = __gui_ico_path__
LOGFILE_PATH = __db_logfile_path__
LMS_PATH = __language_models_path__
HT_PATH = f'{__database_location__}ht/'
WK_PATH = f'{__database_location__}wk/'
# -----------------------------------------------------------


def dbgui() -> None:
    root_node = Tk()
    MainWindow(root_node)
    root_node.iconbitmap(ICO_LOCATION)
    root_node.configure()
    root_node.mainloop()
    _endt = f'\n[{time.ctime()}]\t\tFinishing the current DB session.\n\n'
    with open(LOGFILE_PATH, 'a', encoding='utf-8') as file:
        file.writelines(_endt)
    return
    
    
class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("DATABASE GENERATOR")
        self.master.geometry('690x420')
        self.master.minsize(690, 420)
        self.master.maxsize(690, 420)
        self.colors = ColorStyles
        # -------------------------------------------------------------------------------------------------------------
        #                       VARIABLES
        # -------------------------------------------------------------------------------------------------------------
        # Logfile variables:
        self.log_ptr = 0
        self.log_color = []
        self.log_cat2color = {'Info': self.colors.green,
                              'Warning': self.colors.green,
                              'Error': self.colors.red,
                              'Note': self.colors.green}
        # Generators:
        self.htgenerator = None
        self.wkgenerator = None
        # -------------------------------------------------------------------------------------------------------------
        #                       HYPERTRAINING GENERATOR
        # -------------------------------------------------------------------------------------------------------------
        self.hypertrain_lf = LabelFrame(self.master, width=340, height=200)
        self.hypertrain_lf.place(x=5, y=5)

        self.title_ht_label = Label(self.hypertrain_lf, text='HyperTraining Set Generator', font='Helvetica 12 bold')
        self.title_ht_label.place(x=50, y=0)

        self.tput_label = Label(self.hypertrain_lf, text='Throughput: ')
        self.tput_label.place(x=5, y=30)

        self.noise_label = Label(self.hypertrain_lf, text='Noise: ', font='Helvetica 10 bold')
        self.noise_label.place(x=10, y=55)

        self.noise_label_awgn = Label(self.hypertrain_lf, text='AWGN\t:  mean\t\tvar')
        self.noise_label_awgn.place(x=60, y=55)

        self.noise_label_off = Label(self.hypertrain_lf, text='Offset\t:  mean\t\tvar')
        self.noise_label_off.place(x=60, y=75)

        self.clustersize_label = Label(self.hypertrain_lf, text='Cluser size: ', font='Helvetica 10 bold')
        self.clustersize_label.place(x=10, y=105)
        self.clustersize_label2 = Label(self.hypertrain_lf, text='mean\t             var')
        self.clustersize_label2.place(x=117, y=105)

        self.nomatrix_label = Label(self.hypertrain_lf, text='Number of matrix: ', font='Helvetica 10 bold')
        self.nomatrix_label.place(x=10, y=135)

        self.nomatrix_label = Label(self.hypertrain_lf, text='Name: ')
        self.nomatrix_label.place(x=200, y=135)

        self.train_label = Label(self.hypertrain_lf, text='Train: \t           %', font='Helvetica 9 bold')
        self.train_label.place(x=5, y=165)
        self.validation_label = Label(self.hypertrain_lf, text='Validation: \t  %', font='Helvetica 9 bold')
        self.validation_label.place(x=150, y=165)

        self.train_entry = Entry(self.hypertrain_lf, width=7)
        self.train_entry.place(x=45, y=165)
        self.train_entry.insert(-1, '60')

        self.validation_entry = Entry(self.hypertrain_lf, width=7)
        self.validation_entry.place(x=220, y=165)
        self.validation_entry.insert(-1, '20')

        self.tput_entry = Entry(self.hypertrain_lf, width=7)
        self.tput_entry.place(x=80, y=30)
        self.tput_entry.insert(-1, '32')

        self.awgnmean_entry = Entry(self.hypertrain_lf, width=7)
        self.awgnmean_entry.place(x=155, y=55)
        self.awgnmean_entry.insert(-1, '0')

        self.awgnvar_entry = Entry(self.hypertrain_lf, width=7)
        self.awgnvar_entry.place(x=230, y=55)
        self.awgnvar_entry.insert(-1, '0')

        self.offsetmean_entry = Entry(self.hypertrain_lf, width=7)
        self.offsetmean_entry.place(x=155, y=75)
        self.offsetmean_entry.insert(-1, '0')

        self.offsetvar_entry = Entry(self.hypertrain_lf, width=7)
        self.offsetvar_entry.place(x=230, y=75)
        self.offsetvar_entry.insert(-1, '0')

        self.clustermean_entry = Entry(self.hypertrain_lf, width=7)
        self.clustermean_entry.place(x=155, y=105)
        self.clustermean_entry.insert(-1, '5.5')

        self.clustervar_entry = Entry(self.hypertrain_lf, width=7)
        self.clustervar_entry.place(x=230, y=105)
        self.clustervar_entry.insert(-1, '3.0')

        self.numberofmatrix_entry = Entry(self.hypertrain_lf, width=8)
        self.numberofmatrix_entry.place(x=135, y=137)
        self.numberofmatrix_entry.insert(-1, '1024')

        self.name_entry = Entry(self.hypertrain_lf, width=11)
        self.name_entry.place(x=240, y=137)
        self.name_entry.insert(-1, 'test_ht')

        self.generate_ht_button = HoverButton(self.hypertrain_lf, text='Generate dataset', command=self.generate_ht,
                                              width=20, bg=self.colors.blue)
        self.generate_ht_button.place(x=165, y=25)
        # -------------------------------------------------------------------------------------------------------------
        #                       WIKIREAD GENERATOR
        # -------------------------------------------------------------------------------------------------------------
        self.wikiread_lf = LabelFrame(self.master, width=340, height=200)
        self.wikiread_lf.place(x=345, y=5)

        self.title_wr = Label(self.wikiread_lf, text='WikiRead Set Generator', font='Helvetica 12 bold')
        self.title_wr.place(x=70, y=0)

        self.slm_select_label = Label(self.wikiread_lf, text='Select a Language Model:', font='Helvetica 9 bold')
        self.slm_select_label.place(x=5, y=30)
        self.slm_download_label = Label(self.wikiread_lf, text='Download a Language Model:', font='Helvetica 9')
        self.slm_download_label.place(x=5, y=55)

        self.model_list = []
        with open(__language_models_path__, 'r', encoding='utf-8') as file:
            for line in file:
                self.model_list.append(line.replace('\n', ''))
        self.slm_select = ttk.Combobox(self.wikiread_lf, width=25, state='readonly')
        self.slm_select.place(x=160, y=30)
        self.slm_select["values"] = self.model_list
        self.slm_select.set(self.model_list[0])

        self.tput_label_wk = Label(self.wikiread_lf, text='Throughput: ')
        self.tput_label_wk.place(x=5, y=80)

        self.tput_entry_wk = Entry(self.wikiread_lf, width=8)
        self.tput_entry_wk.place(x=110, y=80)
        self.tput_entry_wk.insert(-1, '256')

        self.nomatrix_label_wk = Label(self.wikiread_lf, text='Number of matrix: ')
        self.nomatrix_label_wk.place(x=5, y=105)

        self.numberofmatrix_entry_wk = Entry(self.wikiread_lf, width=8)
        self.numberofmatrix_entry_wk.place(x=110, y=105)
        self.numberofmatrix_entry_wk.insert(-1, '1024')

        self.train_label_wk = Label(self.wikiread_lf, text='Train: \t           %', font='Helvetica 9 bold')
        self.train_label_wk.place(x=5, y=165)
        self.validation_label_wk = Label(self.wikiread_lf, text='Validation: \t  %', font='Helvetica 9 bold')
        self.validation_label_wk.place(x=150, y=165)
        self.train_entry_wk = Entry(self.wikiread_lf, width=7)
        self.train_entry_wk.place(x=45, y=165)
        self.train_entry_wk.insert(-1, '60')
        self.validation_entry_wk = Entry(self.wikiread_lf, width=7)
        self.validation_entry_wk.place(x=220, y=165)
        self.validation_entry_wk.insert(-1, '20')

        self.nomatrix_label_wk = Label(self.wikiread_lf, text='Name: ')
        self.nomatrix_label_wk.place(x=5, y=130)
        self.name_entry_wk = Entry(self.wikiread_lf, width=15)
        self.name_entry_wk.place(x=68, y=130)
        self.name_entry_wk.insert(-1, 'test_wk')

        self.min_words = Label(self.wikiread_lf, text='Min. words:')
        self.min_words.place(x=170, y=105)

        self.min_works_entry_wk = Entry(self.wikiread_lf, width=7)
        self.min_works_entry_wk.place(x=240, y=105)
        self.min_works_entry_wk.insert(-1, '40')

        self.download_entry = Entry(self.wikiread_lf, width=25)
        self.download_entry.place(x=175, y=55)
        self.download_entry.insert(-1, '')

        self.generate_wk_button = HoverButton(self.wikiread_lf, text='Generate dataset', command=self.generate_wk,
                                              width=20, bg=self.colors.blue)
        self.generate_wk_button.place(x=175, y=130)

        self.download_button = HoverButton(self.wikiread_lf, text='Download model', command=self.download_model,
                                           width=20, bg=self.colors.red)
        self.download_button.place(x=177, y=75)
        # -------------------------------------------------------------------------------------------------------------
        #                       SCROLLED    TEXT
        # -------------------------------------------------------------------------------------------------------------
        self.log_text = scrolledtext.ScrolledText(self.master, height=12, width=81, bd=5, bg='black')
        self.log_text.pack()
        self.log_text.place(x=5, y=210)
        self.lowrite('Welcome to the database generator!\n========================================================='
                     '========================\n',
                     cat='Intro')

    # -------------------------------------------------------------------------------------------------------------
    #                       GUI METHODS
    # -------------------------------------------------------------------------------------------------------------
    def generate_ht(self):
        _generator = dict()
        train = self._tonum(self.train_entry.get())
        validation = self._tonum(self.validation_entry.get())
        test = 100 - train - validation
        if test < 0:
            self.lowrite('Train and validation sum must be under 100%.', cat='Error')
        else:
            _generator['path'] = filedialog.asksaveasfilename(filetypes=[('Database Files', '*.db')],
                                                              initialdir=HT_PATH)
            if _generator['path']:
                _generator['path'] = f'{_generator["path"]}.db'
                _generator['distribution'] = (train, validation, test)
                _generator['tput'] = self._tonum(self.tput_entry.get())
                _generator['awgn_m'] = self._tonum(self.awgnmean_entry.get())
                _generator['awgn_v'] = self._tonum(self.awgnvar_entry.get())
                _generator['off_m'] = self._tonum(self.offsetmean_entry.get())
                _generator['off_v'] = self._tonum(self.offsetvar_entry.get())
                _generator['clust_m'] = self._tonum(self.clustermean_entry.get())
                _generator['clust_v'] = self._tonum(self.clustervar_entry.get())
                _generator['number'] = self._tonum(self.numberofmatrix_entry.get())
                _generator['name'] = self.name_entry.get()
                self.lowrite(f'Creating a database with the following parameters:\n{_generator}', cat='Info')

                queue = multiprocessing.Queue()
                generator = Generator(**_generator)

                self.htgenerator = (generator, queue)
                proc = multiprocessing.Process(target=HtGenerator, args=self.htgenerator)
                proc.start()
                msg = queue.get()
                while msg != 'ENDC':
                    if not queue.empty():
                        msg = queue.get()
                        self.lowrite(msg, cat='Info')
                    self.master.update_idletasks()
                    self.master.update()
                proc.join()
                self.lowrite(f'Database {generator["name"]} created sucessfuly.', cat='Info')

    def lowrite(self, _text, cat=None, extra=None):
        # Write in log files and trackers.
        if extra is not None:
            opening = extra
        else:
            opening = 'a'
        with open(LOGFILE_PATH, opening, encoding='utf-8') as logfile:
            if cat is not None and cat != 'Intro':
                text = f'\n<{time.ctime()}>\t[{cat}]:\t{_text}'
            else:
                text = _text

            if cat != 'Intro':
                logfile.writelines(text)
            else:
                logfile.writelines(f'{text}')

            try:
                self.log_color.append(self.log_cat2color[cat])
            except KeyError:
                self.log_color.append(self.colors.green)

            self.log_text.insert(END, text)
            self.log_text.tag_add(str(self.log_ptr), f'{len(self.log_color)}.{self.log_ptr}', END)
            self.log_text.tag_config(str(self.log_ptr), foreground=self.log_color[-1])
            self.log_ptr += len(text)
            self.log_text.see(END)

    @staticmethod
    def _tonum(strg):
        # Translates a string to float or integer.
        if '.' in strg:
            return float(strg)
        else:
            return int(strg)

    def generate_wk(self):
        _generator = dict()
        train = self._tonum(self.train_entry_wk.get())
        validation = self._tonum(self.validation_entry_wk.get())
        test = 100 - train - validation
        if test < 0:
            self.lowrite('Train and validation sum must be under 100%.', cat='Error')
        else:
            _generator['path'] = filedialog.asksaveasfilename(filetypes=[('Database Files', '*.db')],
                                                              initialdir=WK_PATH)
            if _generator['path']:
                _generator['path'] = f'{_generator["path"]}.db'
                _generator['checkpoint_path'] = f'{_generator["path"]}_checkpoint.db'
                _generator['distribution'] = (train, validation, test)
                _generator['tput'] = self._tonum(self.tput_entry_wk.get())
                _generator['number'] = self._tonum(self.numberofmatrix_entry_wk.get())
                _generator['name'] = self.name_entry_wk.get()
                _generator['model_url'] = self.slm_select.get()
                _generator['minimum_words'] = self._tonum(self.min_works_entry_wk.get())
                self.lowrite(f'Creating a database with the following parameters:\n{_generator}', cat='Info')

                queue = multiprocessing.Queue()
                generator = Generator(**_generator)

                self.wkgenerator = (generator, queue)
                proc = multiprocessing.Process(target=WkGenerator, args=self.wkgenerator)
                proc.start()
                msg = queue.get()
                while msg != 'ENDC':
                    if not queue.empty():
                        msg = queue.get()
                        self.lowrite(msg, cat='Info')
                    self.master.update_idletasks()
                    self.master.update()
                proc.join()
                self.lowrite(f'Database {generator["name"]} created sucessfuly.', cat='Info')

    def download_model(self):
        url = self.download_entry.get()
        self.lowrite(f'Downloading the model: {url}.', cat='Info')
        self.master.update_idletasks()
        self.master.update()
        try:
            SentenceTransformer(url)
            if url not in self.model_list:
                self.model_list.append(url)
                with open(__language_models_path__, 'a', encoding='utf-8') as file:
                    file.writelines(f'\n{url}')
                self.slm_select["values"] = self.model_list
                self.lowrite(f'The model has been downloaded with no errors.', cat='Info')
            else:
                self.lowrite(f'The model is already downloaded, check the model list.', cat='Info')
        except Exception as ex:
            self.lowrite(f'The model does not exist: {ex}.', cat='Info')
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
