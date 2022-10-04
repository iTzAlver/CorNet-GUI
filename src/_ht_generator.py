# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import numpy as np


class HtGenerator:
    def __init__(self, generator, lowrite):
        self.options = generator
        self.lowrite = lowrite
        self.segs = []
        self.mtxs = []
        lowrite('Connection with generator sucessful.', cat='Info')
        self.build()

    def build(self):
        for i in range(0, self.options['number']):
            seg, _mtx = self.single_build()
            mtx = self.awgn_off(_mtx)
            self.segs.append(seg)
            self.mtxs.append(mtx)

    def single_build(self):
        segmentation = np.zeros(self.options['tput'], dtype=np.uint8)
        matrix = np.zeros(self.options['tput'], dtype=np.uint8)
        no_splits = np.round(self.options['clust_m'] + self.options['clust_v'] * np.random.random())

        return segmentation, matrix

    def awgn_off(self, _mtx):
        return _mtx

   

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
