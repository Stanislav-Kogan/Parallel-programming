import numpy as np

def decompose_to_LU(a):
    lu_matrix = np.matrix(np.zeros([a.shape[0], a.shape[1]]))
    n = a.shape[0]

    for k in range(n):
        for j in range(k, n):
            lu_matrix[k, j] = a[k, j] - lu_matrix[k, :k] * lu_matrix[:k, j]
        for i in range(k + 1, n):
            lu_matrix[i, k] = (a[i, k] - lu_matrix[i, : k] * lu_matrix[: k, k]) / lu_matrix[k, k]

    return lu_matrix

def get_L(m):
    L = m.copy()
    for i in range(L.shape[0]):
            L[i, i] = 1
            L[i, i+1 :] = 0
    return np.matrix(L)


def get_U(m):
    U = m.copy()
    for i in range(1, U.shape[0]):
        U[i, :i] = 0
    return U

def solve_LU(lu_matrix, b):
    y = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(y.shape[0]):
        y[i, 0] = b[i, 0] - lu_matrix[i, :i] * y[:i]

    x = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(1, x.shape[0] + 1):
        x[-i, 0] = (y[-i] - lu_matrix[-i, -i:] * x[-i:, 0] )/ lu_matrix[-i, -i]

    return x

A = None
y = None
problem_size = None
answer = None
problem_name = "problem_5"''

with open(f"{problem_name}.txt", "r") as inp:
    inp.readline()
    problem_size = (int)(inp.readline())
    print(f"Problem_size: {problem_size}")
    inp.readline()
    str_arr = inp.readline().replace('\n', '').split(',')
    np_arr = np.array(str_arr).astype(float)
    answer = np_arr[:, np.newaxis]
    print(f"Answer:\t\t{'  '.join(str(s[0]) for s in answer.tolist())}")
    inp.readline()
    A = np.identity(problem_size)
    for i in range(problem_size):
        str_arr = inp.readline().replace('\n','').split(',')
        np_arr = np.array(str_arr)
        A[i,:] = np_arr
    inp.readline()
    str_arr = inp.readline().replace('\n','').split(',')
    np_arr = np.array(str_arr).astype(float)
    y = np_arr[:, np.newaxis]

LU = decompose_to_LU(A)
x = solve_LU(LU, y)

print(f"Solution:\t{'  '.join(str(s[0]) for s in x.tolist())}")

with open(f"{problem_name}_solved_python.txt", 'w') as out:
    out.write(','.join(str(s[0]) for s in x.tolist()))
    out.write("\n")
