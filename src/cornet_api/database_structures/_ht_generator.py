# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import multiprocessing
import numpy as np
from ._generator import Generator
from basenet import BaseNetDatabase


# -----------------------------------------------------------
class HtGenerator:
    """The class HtGenerator: Creates a Hipertraining dataset from a generator."""
    def __init__(self, generator: Generator = None, queue: multiprocessing.Queue = None, scale: int = 255, **kwargs):
        if generator is not None:
            self.options: Generator = generator
        else:
            self.options = Generator(**kwargs)
        if not self.options:
            raise ValueError('HtGenerator: The generator is not valid. Check the input parameters.')
        self.segs: list[np.array] = []
        self.mtxs: list[np.array] = []
        self.queue: multiprocessing.Queue = queue
        self.scale: int = scale
        if queue is not None:
            queue.put('The connection with HTGenerator is sucessful.')
        self.build()
        if queue is not None:
            queue.put('ENDC')

    def build(self):
        # Builds the database.
        for i in range(0, self.options['number']):
            seg, _mtx = self._single_build()
            mtx = self._awgn_off(_mtx)
            self.segs.append(seg)
            self.mtxs.append(mtx)
            if self.queue is not None:
                if self.queue.empty():
                    self.queue.put(f'Building HT database {i} / {self.options["number"]}.')
        if self.queue is not None:
            self.queue.put(f'Building HT database {self.options["number"]} / {self.options["number"]}.')

        if self.options['sym']:
            _mtxs = self._sym(self.mtxs)
        else:
            _mtxs = self.mtxs

        _distribution = {'train': self.options['distribution'][0],
                         'val': self.options['distribution'][1],
                         'test': self.options['distribution'][2]}
        thedb = BaseNetDatabase(_mtxs, self.segs, _distribution, name=self.options['name'])
        if self.options['path']:
            thedb.save(self.options['path'])
        return thedb

    def _single_build(self):
        # Builds only a pair of matrix.
        segmentation = np.zeros(self.options['tput'], dtype=np.uint8)
        no_splits = np.round(np.random.normal(self.options['clust_m'], self.options['clust_v']))
        for _ in range(int(abs(no_splits - 1))):
            segmentation = self._insert_1(segmentation)
        segmentation[0] = 1
        return segmentation, self._seg2mat(segmentation)

    def _awgn_off(self, _mtx):
        # Adds offset and AWGN to the matrix.
        mtx = np.copy(_mtx)
        off = self.scale * np.random.normal(self.options['off_m'], self.options['off_v'])
        for row, rowe in enumerate(_mtx):
            for col, element in enumerate(rowe):
                awgn = self.scale * np.random.normal(self.options['awgn_m'], self.options['awgn_v'])
                if element + awgn + off < 0:
                    mtx[row, col] = abs(element + abs(awgn) + off)
                elif element + awgn + off > self.scale:
                    mtx[row, col] = abs(element - abs(awgn))
                else:
                    mtx[row, col] = abs(element + awgn + off)
        return mtx

    @staticmethod
    def _insert_1(seg):
        # Inserts the number of segmentations in the result.
        if sum(seg) == len(seg):
            return seg
        else:
            idx = np.random.randint(len(seg))
            while seg[idx] != 0:
                idx = np.random.randint(len(seg))
            seg[idx] = 1
        return seg

    def _seg2mat(self, seg):
        # Builds the matrix from the segmentation.
        mat = np.zeros((len(seg), len(seg), 1), dtype=np.uint8)
        base = 0
        placix = 0
        for idx, element in enumerate(seg):
            mat[idx, idx] = self.scale
            if element == 0:
                for ix in range(base, placix + base + 1):
                    mat[idx, ix] = np.array([self.scale])
                    mat[ix, idx] = np.array([self.scale])
                placix += 1
            else:
                base = idx
                placix = 0
        return mat

    def _sym(self, mats: np.array):
        _mats = mats.copy()
        # Set of matrix:
        for nm, _mat in enumerate(_mats):
            for nr, row in enumerate(_mat):
                for nc, element in enumerate(row):
                    if nc == nr:
                        _mats[nm][nc, nr, 0] = self.scale
                    if nc < nr:
                        _mats[nm][nc, nr, 0] = element
                    else:
                        break
        return _mats

    def __repr__(self):
        return 'Warning: HtGenerator objects are not designed to be stored. Consider calling only ' \
               'HtGenerator(generator) as only a class build. This object is built with the following Generator:' \
               f'\n{self.options}'
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #