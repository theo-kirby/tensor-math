#TODO: finish Tensor class and replace matrix.py
# implement permute, reduce
# implement zeros instead of auto
# implement views, pretty print

from __future__ import annotations
from typing import Union, Callable
from random import uniform
import copy

Numeric = Union[int, float]
Operand = Union["Tensor", Numeric]

# row major
class Tensor():

    def __init__(self, shape: list|tuple):

        self.dim = len(shape)
        self.shape = tuple(shape)

        self.size = Tensor.cumprod(shape)
        self.stride = Tensor.calculate_stride(shape)

        self.values = [0] * self.size

    def __getitem__(self, index: tuple) -> float:

        offset = self.index_to_offset(index)
        return self.values[offset]

    def __setitem__(self, index: tuple, value: float) -> None:

        offset = self.index_to_offset(index)
        self.values[offset] = value

    def __copy__(self) -> Tensor:

        t = Tensor(self.shape)
        t.values = self.values.copy()

        return t

    def __add__(self, other: Operand) -> Tensor:
        if isinstance(other, Tensor):
            return self.ew_add(other)
        else:
            return self.scalar_add(other)

    def __sub__(self, other: Operand) -> Tensor:
        if isinstance(other, Tensor):
            return self.ew_sub(other)
        else:
            return self.scalar_sub(other)

    def __mul__(self, other: Operand) -> Tensor:
        if isinstance(other, Tensor):
            return self.ew_mul(other)
        else:
            return self.scalar_mul(other)

    def __matmul__(self, other: Operand) -> Tensor:
        if isinstance(other, Tensor):
            return self.mat_mul(other)
    
    def __repr__(self) -> str:

            # internal method for recursive pretty print
            def unflatten(values: list, shape: tuple) -> list:

                # bottom of the recursion; return the actual values
                if len(shape) == 1:
                    return values

                unit_values = Tensor.cumprod(shape[1:]) # number of values per slice

                result = [] # iterate through this dim and collect slices
                for index in range(shape[0]):

                    # start at the first value of this slice
                    # end at the end of the slice (start + slice size)
                    start = index * unit_values 
                    end = start + unit_values 

                    # recurse!
                    unit = unflatten(values[start:end], shape[1:])
                    result.append(unit)

                return result 

            return f"tensor: ({unflatten(self.values, self.shape)})"

    # return a deep copy of the tensor
    def copy(self) -> Tensor:
        return copy.copy(self)
    
    # return a tensor with values set to the result of the function applied to each element
    def map(self, f: Callable) -> Tensor:
        t = self.copy()
        t.values = [f(element) for element in t.values]
        return t
    
    # 1d array of values
    def to_array(self) -> list:
        return self.values

    # fill all values with a scalar value
    def fill(self, value: float):

        for i in range(self.size):
            self.values[i] = value
    
    # fill all values with random values between -1 and 1
    def randomize(self): # return copy instead?
        for i in range(self.size):
            self.values[i] = uniform(-1, 1)

    # return a tensor with values set to these values plus the scalar value
    def scalar_add(self, value: float) -> Tensor:

        t = self.copy()
        t.values = [element + value for element in t.values]

        return t

    # return a tensor with values set to these values minus the scalar value
    def scalar_sub(self, value: float) -> Tensor:

        t = self.copy()
        t.values = [element - value for element in t.values]

        return t

    # return a tensor with values set to these values multiplied by the scalar value
    def scalar_mul(self, value: float) -> Tensor:

        t = self.copy()
        t.values = [element * value for element in t.values]

        return t

    # return a tensor with values set to these values added element wise to the other tensors values
    def ew_add(self, other: Tensor) -> Tensor:

        assert self.shape == other.shape

        t = self.copy()

        values = []
        for index in range(len(t.values)):
            values.append(t.values[index] + other.values[index])

        t.values = values
        return t

    # return a tensor with values set to these values subtracted element wise from the other tensors values
    def ew_sub(self, other: Tensor) -> Tensor:

        assert self.shape == other.shape

        t = self.copy()

        values = []
        for index in range(len(t.values)):
            values.append(t.values[index] - other.values[index])

        t.values = values
        return t

    # return a tensor with values set to these values multiplied element wise by the other tensors values
    def ew_mul(self, other: Tensor) -> Tensor:
        #hadmard prod
        assert self.shape == other.shape

        t = self.copy()

        values = []
        for index in range(len(t.values)):
            values.append(t.values[index] * other.values[index])

        t.values = values
        return t

    # TODO: implement batch / stacked matmul
    # return a tensor with values set to these values multiplied matrix wise by the other tensors values 
    def mat_mul(self, other: Tensor) -> Tensor:

        self_rows = self.shape[0]
        self_cols = self.shape[1]
        other_rows = other.shape[0]
        other_cols = other.shape[1]

        # can only matmul 2d matricies where cols of a = rows of b
        assert isinstance(other, Tensor)
        assert self.dim == 2 and other.dim == 2
        assert self_cols == other_rows

        t = Tensor((self_rows, other_cols))

        for row in range(self_rows):
            for col in range(other_cols):

                s = 0
                for k in range(self_cols): # shared dim

                    s_idx = row * self_cols + k # self.values[row, k]
                    o_idx = k * other_cols + col # other.values[k, col]

                    s += self.values[s_idx] * other.values[o_idx]

                # result[row, col]
                t.values[row * other_cols + col] = s

        return t

    # 2d transpose (invert the tensor shape)
    def transpose(self):

        # can only transpose matricies
        assert self.dim == 2

        rows, cols = self.shape
        t = Tensor((cols, rows)) # not copy because its opposite shape

        for row in range(rows):
            for col in range(cols):
                t[col, row] = self[row, col]
        
        return t
    
    def permute(self, order: tuple)-> Tensor:
        pass

    # return the flat index (offset) of an n dimensional index
    # nd index (t[i][j]...[n]), flat index (offset: (self.values[n]))
    def index_to_offset(self, index: tuple) -> int:

        # the offset of any given index is equal to
        # the sum of the positions and their coresponding strides
        o = 0
        for i, s in zip(index, self.stride):
            o += i*s

        return o

    @staticmethod # get the stride for any shape 
    def calculate_stride(shape: tuple) -> tuple:

        s = 1
        stride = []

        # iterate backwards through shape, first adding 1, then multiplying by prev stride
        for dimension in reversed(shape):
            stride.append(s)
            s *= dimension

        # return stride in the right oreder
        return tuple(reversed(stride))

    @staticmethod
    def cumprod(l: list|tuple) -> float:
        r = 1
        for e in l:
            r *= e 
        return r 
    
    @staticmethod
    # init a tensor from a list of values
    def from_array(array: list) -> Tensor:
        t = Tensor((len(array), 1))
        t.values = array
        return t



