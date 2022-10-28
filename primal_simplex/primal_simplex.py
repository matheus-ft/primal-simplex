import primal_simplex.utils as ut
import numpy as np


def solve(
    c_input: list[float],
    b_input: list[float],
    A_input: list[list[float]],
    max_iterations=100,
):
    c = ut.vector(c_input)
    b = ut.vector(b_input)
    A = ut.matrix(A_input)
    feasible, basic_indexes, nonbasic_indexes = _find_base(A, b)
    if not feasible:
        return "This problem is infeasible"

    for _ in range(max_iterations):
        success, x, z = _simplex(c, b, A, basic_indexes, nonbasic_indexes)
        if success:
            return x, z
    raise Exception("Not enough iterations. Crank that number up, c'mon!")


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
        return True, "This problem has no finite solution", -np.inf

    B = A(basic_indexes)
    N = A(nonbasic_indexes)
    c_B = c(basic_indexes)
    c_N = c(nonbasic_indexes)

    x_B = ut.solve_system(B, b)

    _lambda = ut.solve_system(B.T, c_B)

    # transform non-basic costs into relative costs
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
    epsilon = np.inf  # just initializations
    p = k
    for i in range(len(y)):
        if y[i] <= 0:  # garanteed that at least one won't trigger
            continue
        r = x_B[i] / y[i]
        if r < epsilon:
            epsilon = r
            p = i

    basic_indexes[p], nonbasic_indexes[k] = nonbasic_indexes[k], basic_indexes[p]
    return False, None, None


def _find_base(A: ut.matrix, b: ut.vector):
    m, n = A.shape

    # making b >= 0
    for i in range(m):
        if b[i] < 0:
            b[i] *= -1  # this changes them inplace
            A[i] *= -1

    # adding m artificial varibles
    I = np.identity(m)
    M = A.extended_by(I)
    c = ut.zeros(n).extended_by(ut.ones(m))
    basic_indexes = list(range(n, n + m))  # basis will be the identity I_{m}
    nonbasic_indexes = list(range(n))

    # solving first phase
    w = None
    finished = False
    while not finished:
        finished, _, w = _simplex(c, b, M, basic_indexes, nonbasic_indexes)
    feasible = w == 0  # True if all the artificial varibles come out of the basis
    nonbasic_indexes = [
        j for j in nonbasic_indexes if j < n
    ]  # removing the artificial indexes
    return feasible, basic_indexes, nonbasic_indexes
