# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import multiprocessing
import os
import logging
import random
import keras
import ray
import numpy as np
import newsegmentation as ns
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

RAY_ADDRESS = r'ray://192.168.79.101:10001'
DBPATH = r'./ns_db/txt/Julen/'
GTPATH = r'./ns_db/gt/Julen/'
DB_SCOPE = [f'{DBPATH}{path}' for path in os.listdir(DBPATH)]
GT_SCOPE = [f'{GTPATH}{path}' for path in os.listdir(GTPATH)]
LEN_SCOPE = len(DB_SCOPE)
EV_SCOPE = zip(GT_SCOPE, DB_SCOPE)
MODEL_PATH_2NPROG = r'./models/wk_32_2nprog.h5'

GUY_CONSTRAINTS = ((float, (0, 1)), (float, (0, np.inf)), (float, (0, 1)), (float, (0, 1)), (float, (0, 1)))
BANDWIDTH = 0.2


# ----------------------------------------------------------------------------------------------------------------------
def main():
    epoch = 10
    with HarmonySearchStrategy(8, 0, clustering=False) as mes:
        for _ in range(epoch):
            mes.evaluate()
            mes.collect()
            mes.selection()
            mes.crossover()
            mes.mutation()
            print(mes)


def overall_fitness_function(guy: list[float, float, float, float, float]):
    global EV_SCOPE
    error = []
    tdm: float = guy[0]
    gpa: tuple = (guy[1], guy[2])
    sdm: tuple = (32, guy[3])
    lcm: tuple = (guy[4],)
    for gt, db in EV_SCOPE:
        cornet_segmentation = MySegmentation(db, tdm=tdm, gpa=gpa, sdm=sdm, lcm=lcm)
        eva = cornet_segmentation.evaluate(gt)
        local_error = (1 - eva['F1']) + eva['WD']
        error.append(local_error)
    return sum(error) / len(error)


def random_fitness_function(guy: list[float, float, float, float, float]):
    global EV_SCOPE, LEN_SCOPE
    tdm: float = guy[0]
    gpa: tuple = (guy[1], guy[2])
    sdm: tuple = (32, guy[3])
    lcm: tuple = (guy[4],)
    idx = random.randint(0, LEN_SCOPE - 1)
    db = DB_SCOPE[idx]
    gt = GT_SCOPE[idx]
    cornet_segmentation = MySegmentation(db, tdm=tdm, gpa=gpa, sdm=sdm, lcm=lcm)
    eva = cornet_segmentation.evaluate(gt)
    error = (1 - eva['F1']) + eva['WD']
    return error


# ----------------------------------------------------------------------------------------------------------------------
class HarmonySearchStrategy:
    def __init__(self, number_of_guys_core, number_of_guys_external, random_performance=False, clustering=True,
                 bw: float = BANDWIDTH, memory_level: int = 20):
        if clustering:
            ray.init(RAY_ADDRESS, runtime_env={'working_dir': '.',
                                               'pip': ['numpy', 'newsegmentation', 'keras', 'tensorflow'],
                                               'excludes': [r'C:\Users\ialve\Desktop\CorNet-GUI\test\ns\models'
                                                            r'\wk_32_conv2d.h5']})
            # self._fr_overall = ray.remote(overall_fitness_function)
            # self._fr_random = ray.remote(random_fitness_function)
            logging.warning(f'[+] Connected to {RAY_ADDRESS}.')

        self.guys = np.random.random((memory_level + (number_of_guys_core + number_of_guys_external),
                                      len(GUY_CONSTRAINTS)))
        self.perf = np.ones(memory_level + (number_of_guys_core + number_of_guys_external), dtype=np.float32)
        self.breakout = number_of_guys_core
        self.constraints = GUY_CONSTRAINTS
        self.bw = np.sqrt(bw)

        self._clustering = clustering
        self._rp = random_performance
        self.memory_level = memory_level

        self._q: list[multiprocessing.Queue] = []
        self._p: list[multiprocessing.Process] = []
        self._futures: list = []

    def evaluate(self):
        self._q = []
        self._p = []
        for guy in self.guys[self.memory_level: self.memory_level + self.breakout]:
            # MP overload.
            self._q.append(multiprocessing.Queue())
            self._p.append(multiprocessing.Process(target=self.fit_guy, args=(guy, self._rp, self._q[-1])))
            self._p[-1].start()

        for guy in self.guys[self.memory_level + self.breakout:]:
            # TODO: Ray overload.
            pass
        return self

    def collect(self):
        query_errors = np.ones(len(self.guys), dtype=np.float32)
        returned = 0
        while returned < self.breakout:
            for nq, queue in enumerate(self._q):
                if not queue.empty():
                    query_errors[nq] = queue.get()
                    returned += 1
        # TODO: RECOVER FROM RAY.
        self.perf[self.memory_level:] = query_errors
        return self

    def crossover(self):
        return self

    def mutation(self):
        for n_guy, guy in enumerate(self.guys):
            for nf, feature in guy:
                this_feat = feature + self.bw * np.random.randn()
                if this_feat < self.constraints[nf][1][0]:
                    this_feat = self.constraints[nf][1][0]
                elif this_feat > self.constraints[nf][1][1]:
                    this_feat = self.constraints[nf][1][1]
                self.guys[n_guy, nf] = this_feat
        return self

    def selection(self):
        eval_positions = self.perf.argsort()
        guys = self.guys[eval_positions]
        self.guys = guys
        self.perf = self.perf[eval_positions]
        return self

    # Fit functions.
    @staticmethod
    def fit_guy(guy, rp, q):
        if rp:
            error = random_fitness_function(guy)
        else:
            error = overall_fitness_function(guy)
        q.put(error)
        return error

    # @staticmethod TODO
    # def fitr_guy(guy, rp):
    #     os.environ["CUDA_VISIBLE_DEVICES"] = ""
    #     if rp:
    #         error_future = self._fr_random.remote(guy)
    #     else:
    #         error_future = self._fr_overall.remote(guy)
    #     return error_future

    # Builtins methods.
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ray.shutdown()
        logging.warning(f'[+] Disconnected from {RAY_ADDRESS}.')

    def __repr__(self):
        min_idx_ = self.perf.argmin()
        min_ = min(self.perf)
        _text_ = f'Guy: {self.guys[min_idx_]}\n Is the best with error of {min_}.'
        return _text_


# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        CLAIMED CLASSES                    #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
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
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        another_strategy = tf.distribute.MirroredStrategy()
        with another_strategy.scope():
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
