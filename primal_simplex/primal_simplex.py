import primal_simplex.utils as ut
import numpy as np


class primal_simplex:
    def __init__(self, c: list[float], b: list[float], A: list[list[float]]) -> None:
        self._c = ut.vector(c)
        self._b = ut.vector(b)
        self._A = ut.matrix(A)

    @staticmethod
    def f(x: ut.vector, c: ut.vector):
        return float(c.T @ x)

    @property
    def c(self) -> ut.vector:
        return self._c

    @property
    def b(self) -> ut.vector:
        return self._b

    @property
    def A(self) -> ut.matrix:
        return self._A

    def solve(self, max_iterations=1000):
        solution = False
        feasible = self._find_base()
        if not feasible:
            return solution, "This problem is infeasible", None

        for _ in range(max_iterations):
            success, x, z = self._simplex()
            if success:
                if z > -np.inf:
                    solution = True
                return solution, x, z
        return solution, "Not enough iterations. Crank that number up, c'mon!", None

    def _simplex(self):
        def solution():
            x = ut.zeros(len(self.c))
            x[self._basic_ind] = x_B
            return True, x, self.f(x_B, c_B)

        def unlimited():
            return True, "This problem has no finite solution", -np.inf

        B = self.A(self._basic_ind)
        N = self.A(self._nonbasic_ind)
        c_B = self.c(self._basic_ind)
        c_N = self.c(self._nonbasic_ind)

        x_B = ut.solve_system(B, self.b)

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

        self._basic_ind[p], self._nonbasic_ind[k] = (
            self._nonbasic_ind[k],
            self._basic_ind[p],
        )
        return False, None, None

    def _find_base(self):
        m, n = self.A.shape

        # making b >= 0
        for i in range(m):
            if self.b[i] < 0:
                self._b[i] *= -1
                self._A[i] *= -1

        # to preserve the original data
        self.oldA = self.A
        self.old_costs = self.c

        # adding m artificial varibles
        I = np.identity(m)
        self._A = self.A.extended_by(I)
        self._c = ut.zeros(n).extended_by(ut.ones(m))
        self._basic_ind = list(range(n, n + m))  # basis will be the identity I_{m}
        self._nonbasic_ind = list(range(n))

        # solving first phase
        w = None
        finished = False
        while not finished:
            finished, _, w = self._simplex()
        feasible = w == 0  # True if all the artificial varibles come out of the basis
        self._nonbasic_ind = [
            j for j in self._nonbasic_ind if j < n
        ]  # removing the artificial indexes
        self._A = self.oldA
        self._c = self.old_costs
        return feasible
