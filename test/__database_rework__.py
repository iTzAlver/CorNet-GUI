# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
import os
from src.database_structures import Database
from basenet import BaseNetDatabase
# -----------------------------------------------------------


if __name__ == '__main__':
    for _path_, _, directory in os.walk('C:/Users/ialve/CorNet/db'):
        if directory:
            if 'unworked' in _path_:
                for file_path in directory:
                    old_db = Database.load(f'{_path_}/{file_path}')
                    x = []
                    x.extend(old_db.dataset.xtrain)
                    x.extend(old_db.dataset.xval)
                    x.extend(old_db.dataset.xtest)
                    y = []
                    y.extend(old_db.dataset.ytrain)
                    y.extend(old_db.dataset.yval)
                    y.extend(old_db.dataset.ytest)
                    old_dist = old_db.distribution
                    new_dist = {'train': old_dist['train'], 'val': old_dist['validation'], 'test': old_dist['test']}
                    new_db = BaseNetDatabase(x, y, distribution=new_dist, name=old_db.name, dtype=('float', 'float'),
                                             rescale=255.0)
                    if 'unsym' in _path_:
                        save_in = f'C:/Users/ialve/CorNet/db/ht/unsym/{old_db.name}.db'
                    else:
                        save_in = f'C:/Users/ialve/CorNet/db/ht/sym/{old_db.name}.db'
                    print(f'Written: {save_in}')
                    new_db.save(save_in)
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
