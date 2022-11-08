# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from src.cornet_api.database_structures.split_utils import split_database
from src.cornet_api.__special__ import __database_location__


# -----------------------------------------------------------
if __name__ == '__main__':
    split_database(f'{__database_location__}/wk/wikipedia_dataset_256.db', 8,
                   f'{__database_location__}/wk/wikipedia_dataset_008.db')
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
