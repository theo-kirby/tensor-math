#TODO: finish Tensor class and replace matrix.py

class Tensor():

    def __init__(self, shape: list):

        self.shape = shape
        self.dim = len(shape)
        self.size = _lprod(shape)
        self.values = [0 for value in range(size)]

        # list product function
        def _lprod(l: list):
            r = 1
            for e in l:
                r *= e 
            return r 
