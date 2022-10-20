# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import os
import newsegmentation as ns
import numpy as np

from src import Model

MAIN_VTT_PATH = './test_files/vtt/Julen/'
MAIN_GT_PATH = './test_files/gt/Julen/'
CACHE_FILE = './cache.json'
MODEL_PATH = '../db/models/conv_n_prog_32_04.h5'
# -----------------------------------------------------------


def test_ns():
    _paths = [f'{MAIN_VTT_PATH}{element}' for element in os.listdir(MAIN_VTT_PATH)]
    _gt_paths = [f'{MAIN_GT_PATH}{element}' for element in os.listdir(MAIN_GT_PATH)]

    for path, gt_path in zip(_paths, _gt_paths):
        news = MySegmentation(path, cache_file=CACHE_FILE)
        news_or = ns.Segmentation(path, cache_file=CACHE_FILE)
        news.plotmtx()
        result = news.evaluate(gt_path, show=True)
        result_or = news_or.evaluate(gt_path, show=True)
        print(f'New: {result}\nvs\nOriginal:{result_or}')


# -----------------------------------------------------------
class MySegmentation(ns.NewsSegmentation):
    @staticmethod
    def _spatial_manager(r, param):
        return my_sdm(r, param)

    @staticmethod
    def _specific_language_model(s):
        return ns.default_slm(s)

    @staticmethod
    def _later_correlation_manager(lm, s, t, param):
        return ns.default_lcm(lm, s, t, param)

    @staticmethod
    def _database_transformation(path, op):
        return ns.default_dbt(path, op)


# -----------------------------------------------------------
def my_sdm(original_r, param, tput=32):
    olen = len(original_r)
    if olen <= tput:
        print('Using CNN.')
        model = Model.load(MODEL_PATH)
        r = np.zeros((tput, tput))
        for nr, row in enumerate(original_r):
            for nc, element in enumerate(row):
                r[nr, nc] = 255 * element
        max_norm = max(r[r < 254.9])

        for nr, row in enumerate(r):
            for nc, element in enumerate(row):
                if nc == nr:
                    r[nr, nc] = 255
                else:
                    r[nr, nc] = element * 255 / max_norm

        seg = model.predict([r], expand=True)
        seg = model.threshold(seg[0], th=0.5)
        segs = []
        for element in seg:
            segs.append(int(element))
        segs[0] = 0
        _segs = np.array(segs[:olen])
        _segs = np.where(_segs >= 0.9)[0]
        _sl = _segs.tolist()
        _sl.append(olen)
        return _sl
    else:
        print('Using default.')
        retval = ns.default_sdm(original_r, param)
        return retval


# -----------------------------------------------------------
if __name__ == '__main__':
    test_ns()
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
