# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
from src.structures import Compiler


# -----------------------------------------------------------
def main() -> None:
    mycomp = Compiler(255,
                      layers=['Dense', 'Flatten', 'Conv2D'],
                      shapes=[(180, ), (None, ), (3, 5)],
                      kwds=[['Arg1', 'Arg11'], [None], ['Arg3']],
                      args=[[0, 'Hello'], [None], ['Flattening']],
                      compiler={'ABC': 5, 'XD': 8, 'DFX': 0},
                      devices=[])
    print(mycomp)
# -----------------------------------------------------------
# Main:


if __name__ == '__main__':
    main()

# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
