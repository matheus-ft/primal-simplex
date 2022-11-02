from primal_simplex import linear_problem as PL


while True:
    m = int(input("Number of constraints: m = "))
    n = int(input("Number of variables: n = "))

    c = [float(a) for a in input("Costs vector: c = ").split(" ")]
    while len(c) != n:
        print(f"Oops, looks like c has {len(c)} elements and we should have {n}")
        c = [float(a) for a in input("Costs vector: c = ").split(" ")]

    b = [float(a) for a in input("Resources vector: b = ").split(" ")]
    while len(b) != m:
        print(f"Oops, looks like b has {len(b)} elements and we should have {m}")
        b = [float(a) for a in input("Resources vector: b = ").split(" ")]

    print("Matrix A = [")
    A = [[float(a) for a in input().split()] for _ in range(m)]
    print("]\n")
    while len(A[0]) != n:
        # if another line has less then n columns,then the code will throw an error
        # for mismatching shapes (i.e. dimensions) when converting it to a matrix
        print(f"Oops, looks like A has {len(A[0])} columns and it should have {n}")
        print("Matrix A = [")
        A = [[float(a) for a in input().split()] for _ in range(m)]
        print("]\n")

    max_iterations = input(
        "Specify the maximum number of iterations (0 means the default 1000): "
    )
    try:
        max_iterations = int(max_iterations)
    except:
        max_iterations = 0

    pl = PL(c, b, A)
    pl.solve(max_iterations)
    print("\n", pl, "\n")

    more = (
        True
        if input("Do you want to solve another problem?(y/N) ").lower() == "y"
        else False
    )

    if not more:
        break
    print("\n")
