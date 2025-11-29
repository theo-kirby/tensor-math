from __future__ import annotations

from typing import Callable
from random import uniform 

# current: no static methods yet, only assertion handling 

class Matrix():

	def __init__(self, rows: int, cols: int):

		self.rows: int = rows
		self.cols: int = cols

		# matrix has a number of rows, a number of columns, and a list of lists of floats 
		self.values = [[0 for col in range(self.cols)] for row in range(self.rows)]

	def __repr__(self) -> str: # allow for easy print(matrix)
		return "\n".join(str(row) for row in self.values)

	def __getitem__(self, row) -> list:
		return self.values[row] # enable matrix[x][y] syntax

	def __setitem__(self, row, value) -> None:
		self.values[row] = value


	# perform a deep copy, used by most methods to not mutate state
	def copy(self) -> Matrix:

		result = Matrix(self.rows, self.cols)

		for row in range(self.rows):
			for col in range(self.cols):
				result[row][col] = self[row][col]

		return result
	
	# init random vals
	def randomize(self):

		for row in range(self.rows):
			for col in range(self.cols):
				self[row][col] = uniform(-1, 1)
	
	# init a single dimensional matrix (vector) from an array
	def init_from_array(self, array: list) -> Matrix:

		assert len(array) == self.rows

		result = self.copy()

		for index, element in enumerate(array):
			result[index][0] = element

		return result

	# convert a vector to an array
	def to_array(self) -> list:

		# TODO: implement full flattening instead
		assert self.cols == 1

		return [row[0] for row in self.values]

	# scalar ops for +, -, *
	def scalar_add(self, x: float) -> Matrix:

		result = self.copy()

		for row in range(result.rows):
			for col in range(result.cols):
				result[row][col] += x

		return result

	def scalar_sub(self, x: float) -> Matrix:

		result = self.copy()

		for row in range(result.rows):
			for col in range(result.cols):
				result[row][col] -= x

		return result
	
	def scalar_mul(self, x: float) -> Matrix:

		result = self.copy()

		for row in range(result.rows):
			for col in range(result.cols):
				result[row][col] *= x

		return result

	# element wise ops for +, -, *
	def ew_add(self, matrix: Matrix) -> Matrix:

		assert self.rows == matrix.rows
		assert self.cols == matrix.cols

		result = self.copy()

		for row in range(result.rows):
			for col in range(result.cols):
				result[row][col] += matrix[row][col]
		
		return result

	def ew_sub(self, matrix: Matrix) -> Matrix:

		assert self.rows == matrix.rows
		assert self.cols == matrix.cols

		result = self.copy()

		for row in range(result.rows):
			for col in range(result.cols):
				result[row][col] -= matrix[row][col]

		return result

	def ew_mul(self, matrix: Matrix) -> Matrix:

		assert self.rows == matrix.rows
		assert self.cols == matrix.cols

		result = self.copy()

		for row in range(result.rows):
			for col in range(result.cols):
				result[row][col] *= matrix[row][col]

		return result

	def mat_mul(self, matrix: Matrix) -> Matrix:

		assert self.cols == matrix.rows

		result = Matrix(self.rows, matrix.cols)
		
		for row in range(self.rows):
			for col in range(matrix.cols):
				sum = 0
				for x in range(self.cols):
					sum += self[row][x] * matrix[x][col]

				result[row][col] = sum

		return result
	
	def transpose(self) -> Matrix: # (invert)

		result = Matrix(self.cols, self.rows)

		for row in range(result.rows):
			for col in range(result.cols):
				result[row][col] = self[col][row]

		return result
	
	def map(self, f: Callable) -> Matrix:

		result = self.copy()

		for row in range(result.rows):
			for col in range(result.cols):
				result[row][col] = f(result[row][col])

		return result


	
