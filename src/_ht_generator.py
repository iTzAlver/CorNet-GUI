# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np
from ._database_structure import Database


class HtGenerator:
    def __init__(self, generator, queue):
        self.options = generator
        self.segs = []
        self.mtxs = []
        self.queue = queue
        queue.put('Connection with generator sucessful.')
        self.build()
        thedb = Database((self.mtxs, self.segs), generator)
        thedb.save()
        queue.put('ENDC')

    def build(self):
        for i in range(0, self.options['number']):
            seg, _mtx = self.single_build()
            mtx = self.awgn_off(_mtx)
            self.segs.append(seg)
            self.mtxs.append(mtx)
            if self.queue.empty():
                self.queue.put(f'Building HT database {i} / {self.options["number"]}.')
        self.queue.put(f'Building HT database {self.options["number"]} / {self.options["number"]}.')

    def single_build(self):
        segmentation = np.zeros(self.options['tput'], dtype=np.uint8)
        no_splits = np.round(self.options['clust_m'] + self.options['clust_v'] * np.random.random())
        for _ in range(int(no_splits)):
            segmentation = self._insert_1(segmentation)
        return segmentation, self._seg2mat(segmentation)

    def awgn_off(self, _mtx):
        mtx = np.copy(_mtx)
        off = self.options['off_m'] + self.options['off_v'] * (np.random.random() - 0.5)
        for row, rowe in enumerate(_mtx):
            for col, element in enumerate(rowe):
                awgn = self.options['awgn_m'] + self.options['awgn_v'] * (np.random.random() - 0.5)
                mtx[row, col] = abs(element + awgn + off)
        return mtx

    @staticmethod
    def _insert_1(seg):
        if sum(seg) == len(seg):
            return seg
        else:
            idx = np.random.randint(len(seg))
            while seg[idx] != 0:
                idx = np.random.randint(len(seg))
            seg[idx] = 1
        return seg

    @staticmethod
    def _seg2mat(seg):
        mat = np.zeros((len(seg), len(seg)), dtype=np.uint8)
        base = 0
        placix = 0
        for idx, element in enumerate(seg):
            mat[idx, idx] = 1
            if element == 0:
                for ix in range(base, placix + base + 1):
                    mat[idx, ix] = 1
                    mat[ix, idx] = 1
                placix += 1
            else:
                base = idx
                placix = 0
        return mat
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
