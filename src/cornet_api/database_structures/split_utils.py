# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from basenet import BaseNetDatabase
import numpy as np


# -----------------------------------------------------------
def split_database(db_path: str, target_tput: int, out_path: str) -> BaseNetDatabase:
    """
    This function formats the database to a target throughput.
    :param db_path: Path to the original database.
    :param target_tput: Target througput.
    :param out_path: Path to the target database.
    :return: The database with the target thoughput.
    """
    db = BaseNetDatabase.load(db_path)
    db_tput = len(db.xtrain[0])
    divisions = np.linspace(0, db_tput, int(db_tput / target_tput) + 1)
    target_db = BaseNetDatabase.load(db_path)

    for subset in ['train', 'val', 'test']:
        x = []
        y = []
        x_ = getattr(db, f'x{subset}')
        y_ = getattr(db, f'y{subset}')

        for _x, _y in zip(x_, y_):
            last_index = 0
            for _division in divisions[1:]:
                division = int(_division)
                _x_ = _x[last_index:division, last_index:division]
                _y_ = _y[last_index:division]
                _y_[0] = 1.0
                x.append(_x_)
                y.append(_y_)
                last_index = division
        setattr(target_db, f'x{subset}', np.array(x))
        setattr(target_db, f'y{subset}', np.array(y))

    target_db.size = (len(target_db.xtrain), len(target_db.xval), len(target_db.xtest))
    target_db.save(out_path)
    return target_db
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
