# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                                                           #
#   This file was created by: Alberto Palomo Alonso         #
# Universidad de Alcalá - Escuela Politécnica Superior      #
#                                                           #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
# Import statements:
class Compiler:
    def __init__(self, tput: int, layers: list, shapes: list[tuple], kwds: list[list], args: list[list], compiler: dict,
                 devices: list, verbose: bool = True):
        self.tput = tput
        self.layers = layers
        self.shapes = shapes
        self.args = []
        self.compiler = compiler
        self.compiled = True
        self.devices = devices

        if len(kwds) != len(args) != len(layers) != len(shapes):
            if verbose:
                print('Number of keywords not equal to number of arguments.')
            self.compiled = False
        else:
            for kwd_s, arg_s in zip(kwds, args):
                dik = {}
                if len(kwd_s) != len(arg_s):
                    if verbose:
                        print(f'Number of member of keywords not equalt to the arguments for the memeber: {kwd_s}')
                    self.compiled = False
                else:
                    for kwd, arg in zip(kwd_s, arg_s):
                        if kwd is not None:
                            dik[kwd] = arg
                self.args.append(dik)

    def __repr__(self):
        return f'Compiler with {len(self.layers)} layers, options:\n{self.compiler}'
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
#                        END OF FILE                        #
# - x - x - x - x - x - x - x - x - x - x - x - x - x - x - #
