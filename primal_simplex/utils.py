import numpy as np
import numpy.linalg as linalg


class vector(np.ndarray):
    """2-dimensional numpy array representing a column vector.

    Extends the numpy.ndarray class to be able to more easily handle vectors in R^n.
    Can be created via a list of floats or by either a 1d or a 2d array.

    Parameters
    ----------
    array_like : list[float] | np.ndarray
        iterable from which the vector is created
    """

    def __new__(cls, array_like: list[float] | np.ndarray):
        v = np.array(array_like).reshape((len(array_like), 1))
        v = np.asarray(v, dtype=float).view(cls)
        return v

    def __call__(self, positions: list[int]):
        v = self[positions]
        return vector(v)

    def extended_by(self, other):
        return vector(np.concatenate([self, other]))

    def index(self, value) -> int:
        return list(self).index(value)


class matrix(np.ndarray):
    """2-dimensional numpy array representing a matrix.

    Extends the numpy.ndarray class to be able to build a matrix from:
        a) list of lists

        b) numpy 2d array

        c) list of vectors (or 1d/2d arrays)

    Parameters
    ----------
    array_like : list[list[float]] | np.ndarray | list[vector]
        iterable from which the matrix is created
    """

    def __new__(cls, array_like: list[list[float]] | np.ndarray | list[vector]):
        m = array_like
        if type(m) == np.ndarray:
            if m.ndim != 2:
                raise Exception("Not possible to create a matrix from a non 2d array")
        elif type(m) == list:
            if type(m[0]) == vector:
                m = np.concatenate(m, axis=1)
            else:
                m = np.array(m)
        m = np.asarray(m, dtype=float).view(cls)
        return m

    def __call__(self, columns: list[int]):
        m = [vector(self[:, j]) for j in columns]
        return matrix(m)

    def extended_by(self, other):
        return matrix(np.concatenate([self, other], axis=1))


def zeros(dimension: int):
    return vector(np.zeros(dimension))


def ones(dimension: int):
    return vector(np.ones(dimension))


def identity(dimension: int):
    return matrix(np.identity(dimension))


def solve_system(A: matrix, b: vector):
    x = linalg.solve(A, b)
    return vector(x)
