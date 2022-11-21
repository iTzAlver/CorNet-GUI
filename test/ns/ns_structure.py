# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import newsegmentation as ns
import numpy as np
import keras
import os
import logging
STUDIED_TH_CONV2D = 0.52
STUDIED_TH_2NPROG = 0.28
CURRENT_TH = STUDIED_TH_2NPROG
MODEL_PATH_2NPROG = r'./models/wk_32_2nprog.h5'
DBPATH = r'./ns_db/txt/Julen/'
GTPATH = r'./ns_db/gt/Julen/'
LOG_DUMP = r'./log_dumps.log'

VISUALIZATION = False
# -----------------------------------------------------------


def main():
    print('[+] Connected to Segmentation Test:\n')
    dbs = [f'{DBPATH}{path}' for path in os.listdir(DBPATH)]
    gts = [f'{GTPATH}{path}' for path in os.listdir(GTPATH)]
    for db, gt in zip(dbs, gts):
        original_segmentation = ns.Segmentation(db)
        cornet_segmentation = MySegmentation(db, sdm=(32, CURRENT_TH), lcm=(0.9,))
        os_eva = original_segmentation.evaluate(gt)
        cs_eva = cornet_segmentation.evaluate(gt, show=VISUALIZATION)
        print(f'Segmentation {db}:\nOriginal: {(os_eva["F1"], os_eva["WD"])}\nCorNet: {(cs_eva["F1"], cs_eva["WD"])}')
        with open(LOG_DUMP, 'w+', encoding='utf-8') as file:
            file.writelines(f'Segmentation {db}:\nOriginal: \t{os_eva}\nCorNet: \t{cs_eva}')


# -----------------------------------------------------------
class CornetSDM:
    def __init__(self, r: np.ndarray, t_put: int, th: float, model_path: str):
        self.original_matrix = r
        self.model_path = model_path
        self.model = self._import_model()

        self.x = self._cut_matrix(r, t_put)
        self.y = self.obtain_output()
        self.directives = self.obtain_directives(self.y, th)

        if len(r) > t_put:
            logging.warning(f'OUTBOUNDED MATRIX: The max size of the mesh is {t_put}x{t_put} but the current matrix '
                            f'size is {len(r)}x{len(r)}.')
        elif len(r) < t_put:
            logging.info(f'UNDERBOUNDED MATRIX: The size of the mesh is {t_put}x{t_put} but the current matrix size is '
                         f'{len(r)}x{len(r)}.')

    def _import_model(self):
        return keras.models.load_model(self.model_path)

    @staticmethod
    def _cut_matrix(r, t_put):
        _retmat_ = np.zeros((t_put, t_put), dtype=np.float32)
        for nr, row in enumerate(r):
            for nc, element in enumerate(row):
                if nc < t_put and nr < t_put:
                    if nc == nr:
                        _retmat_[nr, nc] = 1.
                    _retmat_[nr, nc] = element
        _retmat = np.expand_dims(_retmat_, -1)
        retmat = np.expand_dims(_retmat, 0)
        return retmat

    def obtain_output(self):
        model: keras.Model = self.model
        x = self.x
        y = model.predict(x)[0][:len(self.original_matrix)]
        return y

    @staticmethod
    def obtain_directives(y, th):
        directives = list()
        for index, element in enumerate(y[1:]):
            if element >= th:
                directives.append(index + 1)
        directives.append(len(y) - 1)
        return directives


class MySegmentation(ns.NewsSegmentation):
    @staticmethod
    def _spatial_manager(r, param):
        c_sdm = CornetSDM(r, param[0], param[1], model_path=MODEL_PATH_2NPROG)
        return c_sdm.directives

    @staticmethod
    def _specific_language_model(s):
        return ns.default_slm(s)

    @staticmethod
    def _later_correlation_manager(lm, s, t, param):
        return ns.default_lcm(lm, s, t, param)

    @staticmethod
    def _database_transformation(path, op):
        return ns.default_dbt(path, op)


if __name__ == '__main__':
    main()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
