import primal_simplex.utils as ut
import numpy as np


def solve(c: ut.vector, b: ut.vector, A: ut.matrix, max_iterations=100):
    feasible, basic_indexes, nonbasic_indexes = _find_base(c, b, A, max_iterations)
    if not feasible:
        return "This problem is infeasible"

    for _ in range(max_iterations):
        success, x, z = _simplex(c, b, A, basic_indexes, nonbasic_indexes)
        if success:
            return x, z


def _simplex(
    c: ut.vector,
    b: ut.vector,
    A: ut.matrix,
    basic_indexes: list[int],
    nonbasic_indexes: list[int],
):
    def f(x: ut.vector, c_x=c):
        return float(c_x.T @ x)

    def solution():
        x = ut.zeros(len(c))
        x[basic_indexes] = x_B
        return True, x, f(x_B, c_x=c_B)

    def unlimited():
        return True, None, "This problem has no finite solution"

    B = A(basic_indexes)
    N = A(nonbasic_indexes)
    c_B = c(basic_indexes)
    c_N = c(nonbasic_indexes)

    x_B = ut.solve_system(B, b)

    _lambda = ut.solve_system(B.T, c_B)

    for j in range(len(c_N)):
        c_N[j] -= _lambda.T @ N[:, j]

    c_Nk = min(c_N)
    if c_Nk >= 0:
        return solution()
    k = c_N.index(c_Nk)

    a_Nk = ut.vector(N[:, k])
    y = ut.solve_system(B, a_Nk)

    if np.all(y <= 0):
        return unlimited()
    epsilon = np.inf
    p = k  # just an initialization
    for i in range(len(y)):
        if y[i] <= 0:  # at least one won't trigger
            continue
        r = x_B[i] / y[i]
        if r < epsilon:
            epsilon = r
            p = i

    basic_indexes[p], nonbasic_indexes[k] = nonbasic_indexes[k], basic_indexes[p]
    return False, None, None


def _find_base(
    c_x: ut.vector, b_x: ut.vector, A_x: ut.matrix
) -> tuple[bool, list[int], list[int]]:
    A = A_x.copy()
    b = b_x.copy()
    m, n = A.shape
    for i in range(m):
        if b[i] < 0:
            b[i] *= -1
            A[i] *= -1
    y = 0
    for j in range(n):
        if c_x[j] == 0 and np.all(A[:, j] > 0):
            continue
        y += 1
    I = np.identity(y)
    A = A.extend(I)
    c = ut.zeros(len(c_x)).extend(ut.ones(y))
    basic_indexes = list(range(n, n + y))
    nonbasic_indexes = list(range(n))
    _, _, w = _simplex(c, b, A, basic_indexes, nonbasic_indexes)
    if w != 0:
        return False, basic_indexes, nonbasic_indexes
    return True, basic_indexes, nonbasic_indexes
