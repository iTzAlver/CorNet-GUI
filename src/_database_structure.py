# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import random
import pickle
# -----------------------------------------------------------


class _Dataset:
    def __init__(self, xtrain, ytrain, xtest, ytest, xval, yval):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.xval = xval
        self.yval = yval


class Database:
    def __init__(self, setz, generator, dbtype='hypertrain'):
        self.name = generator['name']
        (xtrain, ytrain), (xtest, ytest), (xval, yval) = self._splitdb(setz, generator['distribution'])
        self.dataset = _Dataset(xtrain, ytrain, xtest, ytest, xval, yval)
        self.distribution = {'train': generator['distribution'][0],
                             'validation': generator['distribution'][1],
                             'test': generator['distribution'][2]}
        self.size = (len(self.dataset.xtrain), len(self.dataset.xval), len(self.dataset.xtest))
        self._path = generator['path']
        self.type = dbtype

    @staticmethod
    def _splitdb(setz: list, split: tuple):
        total = len(setz[0])
        xtrain = []
        ytrain = []
        xval = []
        yval = []
        ntrain = round(total * split[0] / 100)
        nval = round(total * split[1] / 100)
        ntest = total - ntrain - nval
        if ntest >= 0:
            for _ in range(ntrain):
                topop = random.randint(0, len(setz[0]) - 1)
                xtrain.append(setz[0].pop(topop))
                ytrain.append(setz[1].pop(topop))
            for _ in range(nval):
                topop = random.randint(0, len(setz[0]) - 1)
                xval.append(setz[0].pop(topop))
                yval.append(setz[1].pop(topop))
            xtest = setz[0]
            ytest = setz[1]
        else:
            raise ValueError('Test size in Database class is too small.')
        return (xtrain, ytrain), (xtest, ytest), (xval, yval)

    def save(self):
        with open(self._path, 'wb') as file:
            pickle.dump(self, file)

    def __repr__(self):
        return f'Database structure with:\n' \
               f'\tTrain: {len(self.dataset.xtrain)}\n' \
               f'\tValidation: {len(self.dataset.xval)}\n'\
               f'\tTest: {len(self.dataset.xtest)}\n' \
               f'{self.distribution}.'


def load_database(path):
    with open(path, 'rb') as file:
        self = pickle.load(file)
    return self
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
