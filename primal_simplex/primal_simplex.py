import primal_simplex.utils as ut
import numpy as np


def solve(c: ut.vector, b: ut.vector, A: ut.matrix, max_iterations=100):
    def f(x: ut.vector, c_x=c):
        return float(c_x.T @ x)

    def solution() -> tuple[ut.vector, float]:
        x = ut.zeros(len(c))
        x[basic_indexes] = x_B
        return x, f(x_B, c_x=c_B)

    def unlimited():
        return "This problem has no finite solution"

    def infeasible():
        return "This problem is infeasible"

    feasible, basic_indexes, nonbasic_indexes = _find_base(A)
    if not feasible:
        return infeasible()

    for _ in range(max_iterations):
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


def _find_base(A: ut.matrix) -> tuple[bool, list[int], list[int]]:
    pass
