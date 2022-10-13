# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import multiprocessing
import numpy as np
from ._database_structure import Database
from ._generator import Generator


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
        thedb = Database((self.mtxs, self.segs), self.options)
        thedb.save()
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

    def _single_build(self):
        # Builds only a pair of matrix.
        segmentation = np.zeros(self.options['tput'], dtype=np.uint8)
        no_splits = np.round(self.options['clust_m'] + self.options['clust_v'] * np.random.random())
        for _ in range(int(no_splits)):
            segmentation = self._insert_1(segmentation)
        return segmentation, self._seg2mat(segmentation)

    def _awgn_off(self, _mtx):
        # Adds offset and AWGN to the matrix.
        mtx = np.copy(_mtx)
        off = self.scale * (self.options['off_m'] + self.options['off_v'] * (np.random.random() - 0.5))
        for row, rowe in enumerate(_mtx):
            for col, element in enumerate(rowe):
                awgn = self.scale * (self.options['awgn_m'] + self.options['awgn_v'] * (np.random.random() - 0.5))
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
            mat[idx, idx] = 1
            if element == 0:
                for ix in range(base, placix + base + 1):
                    mat[idx, ix] = np.array([self.scale])
                    mat[ix, idx] = np.array([self.scale])
                placix += 1
            else:
                base = idx
                placix = 0
        return mat

    def __repr__(self):
        return 'Warning: HtGenerator objects are not designed to be stored. Consider calling only ' \
               'HtGenerator(generator) as only a class build. This object is built with the following Generator:' \
               f'\n{self.options}'
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
