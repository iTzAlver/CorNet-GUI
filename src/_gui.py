# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import os
from tkinter import Tk, LabelFrame, Label, Entry, ttk, END, scrolledtext, filedialog
from ttkwidgets.autocomplete import AutocompleteCombobox
from .utils import HoverButton, ColorStyles, layers_help
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image

from .compiler_structures import Compiler
from .compiler_structures import TYPEOFLOSES, KERAS_LISTOF_TYPEOFLAYERS, PREBUILT_LAYERS, CUSTOM_LOSES
from .compiler_structures import Model
from .database_structures import Database
from ._dbgui import dbgui
from .report_utils import Report
from .__path_to_config__ import PATH_TO_CONFIG


import time
import graphviz as gv
import matplotlib.pyplot as plt
import pickle
import multiprocessing
import json


# -----------------------------------------------------------
try:
    with open(PATH_TO_CONFIG, 'r') as _file:
        cfg = json.load(_file)
        ICO_LOCATION = cfg["path"]["ICO_LOCATION"]
        LOGFILE_PATH = cfg["path"]["LOGFILE_PATH"]
        LMS_PATH = cfg["path"]["LMS_PATH"]
        DRAW_MODEL_PATH = cfg["path"]["DRAW_MODEL_PATH"]
        COMPILER_LIST_LOCATION = cfg["path"]["COMPILER_LIST_LOCATION"]
        MODEL_LOCATION = cfg["path"]["MODEL_LOCATION"]
        DATABASE_LOCATION = cfg["path"]["DATABASE_LOCATION"]
        MODEL_TEMP = cfg["path"]["MODEL_TEMP"]
    del _file
except Exception as _ex:
    print('Traceback: _gui.py. Path to config corrupted, try to run the gui from the proper executable.')
    raise _ex

# -----------------------------------------------------------


def gui() -> None:
    root_node = Tk()
    MainWindow(root_node)
    root_node.iconbitmap(ICO_LOCATION)
    root_node.configure()
    root_node.mainloop()
    _endt = f'\n[{time.ctime()}]\t\tFinishing the current session.\n\n'
    with open(LOGFILE_PATH, 'a', encoding='utf-8') as file:
        file.writelines(_endt)
    return


class MainWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("CORNET")
        self.master.geometry('620x810')
        self.windowsize = '620x810'
        self.colors = ColorStyles
        self.model_list = self._readmodels_slm()
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
        # Tensorflow devices:
        self.devices = Compiler.show_devs()
        self.devices_list = []
        self.devices_role = {}
        self.device_roles = ['None', 'Train', 'Feed']
        self.device_role2num = {'None': 0, 'Train': 1, 'Feed': 2}
        for dev in self.devices:
            self.devices_list.append(dev)
            self.devices_role[dev] = 0
        # Training variables:
        self.throughput = None
        self.running = False
        self.history = []
        self.eval = []
        # Model variables:
        add_pipeline = ['open_pipeline', 'close_pipeline', '']
        self.typeoflayers = KERAS_LISTOF_TYPEOFLAYERS
        self.typeoflayers.extend(PREBUILT_LAYERS)
        self.typeoflayers.extend(add_pipeline)
        self.losses = TYPEOFLOSES
        self.losses.extend(CUSTOM_LOSES)
        self.metrics = ['accuracy', 'mse']
        self.optimizers = ['adadelta', 'adam', 'RMSprop', 'sgd', 'nadam', 'adamax', 'adagrad', 'ftlr']
        self.current_model_list = []
        self.model = None
        # Canvas:
        self.canvas1 = None
        self.toolbar1 = None
        self.canvas2 = None
        self.toolbar2 = None
        # Database variables:
        self.database = None
        # -------------------------------------------------------------------------------------------------------------
        #                       DATABASE FRAME
        # -------------------------------------------------------------------------------------------------------------
        self.database_lf = LabelFrame(self.master, width=220, height=105)
        self.database_lf.place(x=5, y=115)
        self.generate_db_button = HoverButton(self.database_lf,
                                              text='GENERATE CUSTOM DATABASE',
                                              command=self.gui_database,
                                              width=27, bg=self.colors.blue)
        self.generate_db_button.place(x=5, y=5)
        self.import_db_button = HoverButton(self.database_lf,
                                            text='IMPORT DATABASE',
                                            command=self.import_database,
                                            width=27, bg=self.colors.blue)
        self.import_db_button.place(x=5, y=35)

        self.export_model_button = HoverButton(self.database_lf,
                                               text='EXPORT MODEL',
                                               command=self.export_model,
                                               width=12, bg=self.colors.gray)
        self.export_model_button.place(x=5, y=65)

        self.import_model_button = HoverButton(self.database_lf,
                                               text='IMPORT MODEL',
                                               command=self.import_model,
                                               width=12, bg=self.colors.gray)
        self.import_model_button.place(x=110, y=65)
        # -------------------------------------------------------------------------------------------------------------
        #                       LIVEFEED FRAME
        # -------------------------------------------------------------------------------------------------------------
        self.livefeed_lf = LabelFrame(self.master, width=220, height=105)
        self.livefeed_lf.place(x=5, y=225)

        self.livefeed_label = Label(self.livefeed_lf, text='LIVE FEED')
        self.livefeed_label.place(x=75, y=0)

        self.livefeed_batch_label = Label(self.livefeed_lf, text='Batch DB size: ')
        self.livefeed_batch_label.place(x=5, y=25)

        self.livefeed_model_label = Label(self.livefeed_lf, text='Language model: ')
        self.livefeed_model_label.place(x=5, y=45)

        self.feed_button = HoverButton(self.livefeed_lf,
                                       text='START FEED',
                                       command=self.start_feed,
                                       width=12, bg=self.colors.yellow)
        self.feed_button.place(x=5, y=70)
        self.stop_feed_button = HoverButton(self.livefeed_lf,
                                            text='STOP FEED',
                                            command=self.stop_feed,
                                            width=12, bg=self.colors.red)
        self.stop_feed_button.place(x=110, y=70)

        self.livefeed_batch_entry = Entry(self.livefeed_lf, width=12)
        self.livefeed_batch_entry.place(x=120, y=25)
        self.livefeed_batch_entry.insert(-1, '32')
        self.livefeed_model_entry = ttk.Combobox(self.livefeed_lf, width=9, state='readonly')
        self.livefeed_model_entry.place(x=120, y=45)
        self.livefeed_model_entry["values"] = self.model_list
        self.livefeed_model_entry.set(self.model_list[0])
        # -------------------------------------------------------------------------------------------------------------
        #                       DEVICES MANAGEMENT
        # -------------------------------------------------------------------------------------------------------------
        self.devices_role_lf = LabelFrame(self.master, width=220, height=135)
        self.devices_role_lf.place(x=5, y=335)

        self.device_role_label = Label(self.devices_role_lf, text='DEVICE ROLE')
        self.device_role_label.place(x=75, y=0)

        self.role_set_up = HoverButton(self.devices_role_lf,
                                       text='SET UP ROLE',
                                       command=self.set_up_role,
                                       width=27, bg=self.colors.orange)
        self.role_set_up.place(x=5, y=70)

        self.role_listing_buton = HoverButton(self.devices_role_lf,
                                              text='LIST DEVICES',
                                              command=self.list_devices,
                                              width=27, bg=self.colors.yellow)
        self.role_listing_buton.place(x=5, y=100)

        self.device_selection_label = Label(self.devices_role_lf, text='Select device: ')
        self.device_selection_label.place(x=5, y=25)
        self.device_role_label_select = Label(self.devices_role_lf, text='Select role: ')
        self.device_role_label_select.place(x=5, y=45)

        self.device_selection = ttk.Combobox(self.devices_role_lf, width=9, state='readonly')
        self.device_selection.place(x=120, y=25)
        self.device_selection["values"] = self.devices_list
        self.device_selection.set(self.devices_list[0])

        self.device_role_select = ttk.Combobox(self.devices_role_lf, width=9, state='readonly')
        self.device_role_select.place(x=120, y=45)
        self.device_role_select["values"] = self.device_roles
        self.device_role_select.set(self.device_roles[0])
        # -------------------------------------------------------------------------------------------------------------
        #                       LOGTRACKER FRAME
        # -------------------------------------------------------------------------------------------------------------
        self.log_text = scrolledtext.ScrolledText(master, height=6, width=73, bd=5, bg='black')
        self.log_text.pack()
        self.log_text.place(x=5, y=5)
        self.lowrite(f'[{time.ctime()}]\t\tWelcome to CORNET!', cat='Intro')
        self.list_devices()
        # -------------------------------------------------------------------------------------------------------------
        #                       TRAINING FRAME
        # -------------------------------------------------------------------------------------------------------------
        self.training_lf = LabelFrame(self.master, width=220, height=40)
        self.training_lf.place(x=5, y=475)

        self.training_label = Label(self.training_lf, text=f'Throughtput: {self.throughput}')
        self.training_label.place(x=5, y=7)

        self.training_button = HoverButton(self.training_lf,
                                           text='TRAIN',
                                           command=self.train_static,
                                           width=10, bg=self.colors.blue)
        self.training_button.place(x=125, y=5)
        # -------------------------------------------------------------------------------------------------------------
        #                       MONITOR FRAME
        # -------------------------------------------------------------------------------------------------------------
        self.monitor_lf = LabelFrame(self.master, width=374, height=400)
        self.monitor_lf.place(x=228, y=115)

        self.monitor_label = Label(self.monitor_lf, text='COMPILED MODEL')
        self.monitor_label.place(x=140, y=3)

        self.monitor_esq = LabelFrame(self.monitor_lf, width=370, height=370)
        self.monitor_esq.place(x=0, y=26)
        # -------------------------------------------------------------------------------------------------------------
        #                       MODEL FRAME
        # -------------------------------------------------------------------------------------------------------------
        self.model_lf = LabelFrame(self.master, width=597, height=292)
        self.model_lf.place(x=5, y=515)

        self.model_lf_add = LabelFrame(self.model_lf, width=198, height=288)
        self.model_lf_add.place(x=0, y=0)

        self.model_lf_ovw = LabelFrame(self.model_lf, width=392, height=288)
        self.model_lf_ovw.place(x=200, y=0)

        self.type_label = Label(self.model_lf_add, text='Type of layer:')
        self.type_label.place(x=5, y=5)
        self.shape_label = Label(self.model_lf_add, text='Shape:')
        self.shape_label.place(x=5, y=30)
        self.extras_label = Label(self.model_lf_add, text='Extras:')
        self.extras_label.place(x=5, y=55)

        self.layer_selection = AutocompleteCombobox(self.model_lf_add, width=13, state='readonly',
                                                    completevalues=self.typeoflayers)
        self.layer_selection.place(x=90, y=5)
        self.layer_selection["values"] = self.typeoflayers
        self.layer_selection.set(self.typeoflayers[0])
        self.layer_selection.bind("<<ComboboxSelected>>", self._showhelp)

        self.shape_entry = Entry(self.model_lf_add, width=16)
        self.shape_entry.place(x=90, y=30)

        self.extra_name_entry = Entry(self.model_lf_add, width=9)
        self.extra_name_entry.place(x=90, y=55)

        self.extra_value_entry = Entry(self.model_lf_add, width=6)
        self.extra_value_entry.place(x=150, y=55)

        self.add_layer_button = HoverButton(self.model_lf_add,
                                            text='ADD LAYER',
                                            command=self.add_layer,
                                            width=11, bg=self.colors.blue)
        self.add_layer_button.place(x=5, y=85)

        self.model_compile_button = HoverButton(self.model_lf_add,
                                                text='COMPILE',
                                                command=self.compile,
                                                width=11, bg=self.colors.yellow)
        self.model_compile_button.place(x=100, y=85)

        self.model_replace_button = HoverButton(self.model_lf_add,
                                                text='REPLACE',
                                                command=self.replace,
                                                width=11, bg=self.colors.yellow)
        self.model_replace_button.place(x=5, y=150)

        self.model_remove_button = HoverButton(self.model_lf_add,
                                               text='REMOVE',
                                               command=self.remove,
                                               width=11, bg=self.colors.red)
        self.model_remove_button.place(x=100, y=150)

        self.listoflayers = ttk.Combobox(self.model_lf_add, width=27, state='readonly')
        self.listoflayers.place(x=5, y=120)
        self.listoflayers["values"] = self.current_model_list

        self.loss_label = Label(self.model_lf_add, text='Loss:')
        self.loss_label.place(x=5, y=220)

        self.opt_label = Label(self.model_lf_add, text='Optimizer:')
        self.opt_label.place(x=5, y=250)

        self.opt_label = Label(self.model_lf_add, text='COMPILE OPTIONS:')
        self.opt_label.place(x=45, y=185)

        self.loss_selection = ttk.Combobox(self.model_lf_add, width=21, state='readonly')
        self.loss_selection.place(x=40, y=220)
        self.loss_selection["values"] = self.losses
        self.loss_selection.set(self.losses[3])

        self.opt_selection = ttk.Combobox(self.model_lf_add, width=16, state='readonly')
        self.opt_selection.place(x=70, y=250)
        self.opt_selection["values"] = self.optimizers
        self.opt_selection.set(self.optimizers[1])
        # -------------------------------------------------------------------------------------------------------------
        #                       HELP TEXTBOX
        # -------------------------------------------------------------------------------------------------------------
        self.help_text = scrolledtext.ScrolledText(self.master, height=49, width=80, bd=5, bg='black')
        self.help_text.pack()
        self.help_text.place(x=620, y=5)
        self.expander = HoverButton(self.master,
                                    text='<\n>',
                                    command=self._expand,
                                    height=53,
                                    width=1,
                                    bg=self.colors.white)
        self.expander.place(x=602, y=3)
        # -------------------------------------------------------------------------------------------------------------
        #                       LAST CALLS
        # -------------------------------------------------------------------------------------------------------------
        self._read_models()

    # -------------------------------------------------------------------------------------------------------------
    #                       EXTERNAL FUNCTIONS
    # -------------------------------------------------------------------------------------------------------------
    def gui_database(self):
        # Database generator interface.
        self.lowrite(_text='Opening dual interface for database building.', cat='Info')
        p = multiprocessing.Process(target=dbgui)
        p.start()

    def import_database(self):
        # Import the generated database.
        dbpath = filedialog.askopenfilename(filetypes=[('Database files', '*.ht')], initialdir=DATABASE_LOCATION)
        if dbpath:
            database = Database.load(dbpath)
            self.database = database
            self.throughput = len(self.database.dataset.xtrain[0])
            self.training_label.config(text=f'Throughtput: {self.throughput}')
            self.lowrite(_text=f'Database {database.name} loaded.', cat='Info')

    def export_model(self):
        # Export the deep learning model.
        filename = filedialog.asksaveasfilename(filetypes=[('Binary files', '*')], initialdir=COMPILER_LIST_LOCATION)
        if filename:
            if self.model is None:
                if not self.compile():
                    self.lowrite(_text=f'Cannot export the current model: the model is not compilable.',
                                 cat='Info')
                    return
            with open(filename, 'wb') as file:
                pickle.dump(self.current_model_list, file)
            _filename = filename.split('/')[-1]
            self.model.save(f'{MODEL_LOCATION}/{_filename}.h5')
            self._read_models()
            self.lowrite(_text=f'Model exported to {filename}.', cat='Info')
        Report(image_path='../multimedia/render/compiled-model.gv.png',
               latex_path='../temp/latex',
               model=self.model,
               db=self.database,
               history=self.history)

    def import_model(self):
        # Import the deep learning model.
        filename = filedialog.askopenfilename(filetypes=[('Binary files', '*')], initialdir=COMPILER_LIST_LOCATION)
        if filename:
            with open(filename, 'rb') as file:
                self.current_model_list = pickle.load(file)
                self.listoflayers["values"] = [val[0] for val in self.current_model_list]
            self.model = Model.load(f'{MODEL_LOCATION}/{filename.split("/")[-1]}.h5')
            self.lowrite(f'Model summary:\n{self.model.summary}', cat='Info')
            self.draw_scheme()
            self.throughput = self.model.compiler.io_shape[1]
            self.lowrite(_text=f'Imported model {filename}.', cat='Info')

    def start_feed(self):
        # Start feeding. TODO
        self.lowrite(_text='Starting model feeding with custom batch size: {}.', cat='Info')
        pass

    def stop_feed(self):
        # Stop feeding. TODO
        self.lowrite(_text='Real time feeding stopped.', cat='Info')
        pass

    def train_static(self):
        # Training the model with the database.
        if self.running:
            self.running = False
            self.lowrite(_text='Training stopped.', cat='Info')
        else:
            self.running = True
            self.lowrite(_text='Training the current model.', cat='Info')

            bypass = self.model.bypass(MODEL_TEMP)
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=Model.fitmodel,
                                        args=(self.model, self.database, queue),
                                        kwargs={'bypass': bypass})
            p.start()
            while self.running:
                self.master.update_idletasks()
                self.master.update()
                if not queue.empty():
                    history = queue.get()
                    self.history.append(history['loss'])
                    self.eval.append(history['eval'])
                    self._print_history()
            queue.put('MASTER:STOP')
            msg = ''
            while msg != 'SLAVE:STOPPED':
                if not queue.empty():
                    msg = queue.get()
            self.model.bypass(bypass)

    def _compile(self, compiler):
        # Compile the model into self.model.
        try:
            self.model = compiler.compile()
            self.lowrite(f'Model summary:\n{self.model.summary}', cat='Info')
            return True
        except Exception as ex:
            self.lowrite(f'The model has the following errors:\n{ex}', cat='Error')
            return False

    # -------------------------------------------------------------------------------------------------------------
    #                       INTERFACE METHODS
    # -------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _readmodels_slm():
        # Read the SLM models from the .txt file.
        modelist = []
        with open(LMS_PATH, 'r', encoding='utf-8') as file:
            for line in file:
                modelist.append(line)
        return modelist

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

    def set_up_role(self):
        # Set up the device role.
        dev = self.device_selection.get()
        role = self.device_role_select.get()
        self.devices_role[dev] = self.device_role2num[role]
        self.lowrite(_text=f'Device {dev} set up for {role}.', cat='Info')
        pass

    def list_devices(self):
        # List the current devices in the current sesion.
        self.lowrite(f'Current available devices:\n', cat='Info')
        for ix, dev in enumerate(self.devices_list):
            self.lowrite(f'\tDevice {ix}:\t{dev}\t\tRole: {self.device_roles[self.devices_role[dev]]}\n', cat='Intro')

    def add_layer(self, idx=None):
        # Add the layer button.
        typ = self.layer_selection.get()
        shape = self.shape_entry.get().split(', ')

        if shape[0]:
            for index, shap in enumerate(shape):
                shape[index] = self._tonum(shap)
        else:
            shape = ''

        extras = self.extra_name_entry.get()
        values = self.extra_value_entry.get()
        if extras:
            extras = extras.split(', ')
        if values:
            values = values.split(', ')
            for i, value in enumerate(values):
                if value[0] == "\'":
                    values[i] = value.replace('\'', '')
                elif value[0] == '(':
                    values[i] = tuple([self._tonum(val) for val in value.replace('(', '').replace(')', '').split(',')])
                else:
                    values[i] = self._tonum(value)
        if len(extras) != len(values):
            self.lowrite(_text=f'Cannot match {len(extras)} keywords to {len(values)} atributes.', cat='Error')
            return
        else:
            self.lowrite(_text=f'Adding new layer to the model:\n\t\t{typ} {shape}', cat='Info')
            if idx is None:
                self.current_model_list.append((f'{typ} {shape}', (extras, values)))
            else:
                self.current_model_list[idx] = (f'{typ} {shape}', (extras, values))
            self.listoflayers["values"] = [val[0] for val in self.current_model_list]

        self.shape_entry.delete(0, END)
        self.extra_name_entry.delete(0, END)
        self.extra_value_entry.delete(0, END)
        return

    def compile(self):
        # Compile the current model.
        if self.current_model_list:
            self.lowrite(_text='Compiling the current model:\n', cat='Info')
            current_model_list = [f'Input:{self.throughput}']
            current_model_list.extend([val[0] for val in self.current_model_list])
            current_model_list.append(f'Output:{self.throughput}')
            for layer in current_model_list:
                self.lowrite(_text=f'\t\t{layer}\n', cat='Intro')

            compiled_model = self._parse_compiler()
            if self._compile(compiled_model):
                self.lowrite(_text='The model compiled sucessfuly.', cat='Info')
                self.shape_entry.delete(0, END)
                self.extra_name_entry.delete(0, END)
                self.extra_value_entry.delete(0, END)
                self.draw_scheme()
            else:
                self.lowrite(_text='The model compiled with errors.', cat='Info')
        else:
            self.lowrite(_text='Cannot compile the model: it is empty.', cat='Error')
            return False
        return True

    def replace(self):
        # Replace a layer from in target.
        idx = self.listoflayers.current()
        removed = None
        if self.current_model_list:
            removed = self.current_model_list[idx]
            self.add_layer(idx=idx)
        self.listoflayers["values"] = [val[0] for val in self.current_model_list]
        if removed is not None:
            self.lowrite(_text=f'Replaced {removed} from the layer list.', cat='Info')
        else:
            self.lowrite(_text=f'Target not replaced: the list is empty.', cat='Warning')
        return

    def remove(self):
        # Pop a layer from the target.
        idx = self.listoflayers.current()
        removed = None
        if self.current_model_list:
            removed = self.current_model_list.pop(idx)
        self.listoflayers["values"] = [val[0] for val in self.current_model_list]
        if removed is not None:
            self.lowrite(_text=f'Removed {removed} from the layer list.', cat='Info')
        else:
            self.lowrite(_text=f'Target not removed: the list is empty.', cat='Warning')
        return

    def draw_scheme(self, tf_draw=True):
        # Draw the model scheme.
        model_list = self.current_model_list
        if tf_draw:
            self.model.model_print(DRAW_MODEL_PATH)
        else:
            dot = gv.Digraph('compiled-model', comment=f'{time.ctime()}')
            dot.format = 'png'
            dot.node('0', f'Input mesh\n({self.throughput}) x ({self.throughput})', shape='doubleoctagon')
            idx = 0
            for idx, layer in enumerate(model_list):
                if layer[1] != ('', ''):
                    dot.node(f'{idx + 1}', f'{layer[0]}\n{layer[1]}', shape='box')
                else:
                    dot.node(f'{idx + 1}', f'{layer[0]}', shape='box')
                dot.edge(f'{idx}', f'{idx + 1}')
            dot.node(f'{idx + 2}', f'Output mesh\n({self.throughput})', shape='doubleoctagon')
            dot.edge(f'{idx + 1}', f'{idx + 2}')

            try:
                dot.render(directory=DRAW_MODEL_PATH)
            except Exception as ex:
                self.lowrite('Graphviz is not installed on this machine, please install Graphviz in your current'
                             f'machine and add Graphviz to the PATH system variable: {ex}', cat='Error')

        png = Image.open(f'{DRAW_MODEL_PATH}/compiled-model.gv.png')
        myfig = plt.figure(figsize=(4.87, 4.32), dpi=75)
        plt.imshow(png.convert('RGB'))
        plt.axis('off')
        if self.canvas1 is not None:
            self.canvas1.get_tk_widget().pack_forget()
            self.toolbar1.destroy()
        self.canvas1 = FigureCanvasTkAgg(myfig, master=self.monitor_esq)
        self.canvas1.draw()
        self.toolbar1 = NavigationToolbar2Tk(self.canvas1, self.monitor_esq)
        self.toolbar1.update()
        self.canvas1.get_tk_widget().pack()
        self.lowrite(_text=f'Scheme drawn sucessfuly.', cat='Info')

    def _parse_compiler(self):
        # Parses the current model list into the compiler.
        model_list = self.current_model_list
        layers = []
        shapes = []
        kwds = []
        args = []
        compiler = {'loss': self.loss_selection.get(),
                    'optimizer': self.opt_selection.get(),
                    'metrics': self.metrics}
        devices = self.devices_role
        for layer in model_list:
            ls = layer[0].replace(',', '').replace('[', '').replace(']', '').split(' ')
            layers.append(ls[0])
            if ls[1] != '':
                _ls = []
                for __ls in ls[1:]:
                    _ls.append(int(__ls))
                shapes.append(tuple(_ls))
            else:
                shapes.append((None, ))

            kwd = layer[1][0]
            arg = layer[1][1]
            if kwd:
                kwds.append(kwd)
                args.append(arg)
            else:
                kwds.append([None])
                args.append([None])

        mycompiler = Compiler(io_shape=((self.throughput, self.throughput, 1), self.throughput), layers=layers,
                              shapes=shapes, kwds=kwds, args=args, compiler=compiler, devices=devices)

        if mycompiler.is_valid:
            return mycompiler
        else:
            self.lowrite('The model compiled with errors... Check your parameters again.', cat='Error')
            return None

    @staticmethod
    def _tonum(strg):
        # Translates a string to float or integer.
        if '.' in strg:
            return float(strg)
        else:
            return int(strg)

    def _showhelp(self, event):
        # Shows the help for each layer selected.
        help_msg = layers_help(self.layer_selection.get())
        self.help_text.delete('1.0', END)
        self.help_text.insert(END, help_msg)
        self.help_text.tag_add('0', f'0.1', END)
        self.help_text.tag_config('0', foreground=self.colors.green)
        self.help_text.see('0.0')
        return event

    def _expand(self):
        # Resizes the window in order to hide the information about the layers.
        if self.windowsize == '1290x810':
            self.master.geometry('620x810')
            self.windowsize = '620x810'
        elif self.windowsize == '620x810':
            self.master.geometry('1290x810')
            self.windowsize = '1290x810'

    def _read_models(self):
        # This function reads the current custom models and updates the model list.
        cmodels = [cmodel[:-3] for cmodel in os.listdir(MODEL_LOCATION)]
        self.typeoflayers = KERAS_LISTOF_TYPEOFLAYERS
        self.typeoflayers.extend(cmodels)
        self.layer_selection["values"] = self.typeoflayers

    def _print_history(self):
        myfig = plt.figure(figsize=(4.86, 3), dpi=80)
        plt.plot(self.history, 'b', label='Loss')
        plt.plot(self.eval, 'r', label='WD')
        plt.legend()
        plt.title('Learning curve')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-')
        if self.canvas2 is not None:
            self.canvas2.get_tk_widget().pack_forget()
            self.toolbar2.destroy()
        self.canvas2 = FigureCanvasTkAgg(myfig, master=self.model_lf_ovw)
        self.canvas2.draw()
        self.toolbar2 = NavigationToolbar2Tk(self.canvas2, self.model_lf_ovw)
        self.toolbar2.update()
        self.canvas2.get_tk_widget().pack()
        plt.close(myfig)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
