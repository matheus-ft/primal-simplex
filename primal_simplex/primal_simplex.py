from . import utils as ut
import numpy as np


class primal_simplex:
    def __init__(self, c: list[float], b: list[float], A: list[list[float]]) -> None:
        self._c = ut.vector(c)
        self._b = ut.vector(b)
        self._A = ut.matrix(A)
        self._problem: str = "This problem has not been solved yet."

    @property
    def basic(self) -> list[int]:
        return self._basic

    @property
    def nonbasic(self) -> list[int]:
        return self._nonbasic

    @property
    def decision_var(self) -> ut.vector | None:
        return self._decision_var

    @property
    def optimal_value(self) -> float | None:
        return self._optimal_value

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

    def __repr__(self) -> str:
        return self._problem

    def solve(self, max_iterations=1000) -> None:
        self._decision_var = None
        self._optimal_value = np.inf
        feasible = self._find_base()
        if not feasible:
            self._problem = "This problem is infeasible."
            return

        for i in range(max_iterations):
            success, x, z = self._simplex()
            if success:
                self._optimal_value = z
                if z > -np.inf:
                    self._decision_var = x
                    self._problem = f"The optimal solution is x = {[round(float(v), 2) for v in x]} with f(x) = {round(z, 2)}, found in {i+1} iterations"
                else:
                    self._problem = "This problem has no finite solution."
                return
        self._problem = f"{max_iterations} are not enough iterations."
        return

    def _simplex(self) -> tuple[bool, ut.vector, float]:
        done = True  # to make a tiny bit more readable

        def solution():
            x = ut.zeros(len(self.c))
            x[self.basic] = x_B
            return done, x, self.f(x_B, c_B)

        def unlimited():
            return done, ut.inf(len(self.c), -1), -np.inf

        # partitions
        B = self.A(self.basic)
        N = self.A(self.nonbasic)
        c_B = self.c(self.basic)
        c_N = self.c(self.nonbasic)

        x_B = ut.solve_system(B, self.b)  # basic solution

        _lambda = ut.solve_system(B.T, c_B)  # simplex multiplier

        # transform non-basic costs into relative costs
        for j in range(len(c_N)):
            c_N[j] -= _lambda.T @ N[:, j]

        c_Nk = min(c_N)
        if c_Nk >= 0:  # optimality test
            return solution()
        k = c_N.index(c_Nk)
        nonbasic_k = self.nonbasic[k]  # index of the variable coming into the basis

        a_Nk = ut.vector(N[:, k])
        y = ut.solve_system(B, a_Nk)  # simplex direction

        if np.all(y <= 0):
            return unlimited()
        epsilon, p = np.inf, k  # just initializations
        for i in range(len(y)):
            if y[i] <= 0:  # garanteed that at least one won't trigger
                continue
            r = x_B[i] / y[i]
            if r < epsilon:
                epsilon = r  # simplex step
                p = i
        basic_p = self.basic[p]  # index of the variable getting out of the basis

        self._basic[p], self._nonbasic[k] = nonbasic_k, basic_p  # swap columns
        return not done, ut.inf(len(self.c)), np.inf

    def _find_base(self) -> bool:
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
        self._basic = list(range(n, n + m))  # first basis will be the identity I_{m}
        self._nonbasic = list(range(n))

        # solving first phase
        w = None
        finished = False
        while not finished:
            finished, _, w = self._simplex()
        feasible = w == 0  # True if all the artificial varibles came out of the basis

        # removing the artificial stuff
        self._nonbasic = [j for j in self.nonbasic if j < n]
        self._A = self.oldA
        self._c = self.old_costs
        return feasible
