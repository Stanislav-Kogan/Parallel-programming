import random
import numpy as np
import math

problem_name = 'problem'
problem_size = 8000
steps = (int)(math.sqrt(problem_size)*1.3)

answers = np.random.randint(1, 11, size=(problem_size))
y = answers.copy()

A = np.identity(problem_size)

A2 = A.copy()
y2 = y.copy()

for n in range (steps):
    i = random.randint(0, problem_size - 1)
    j = random.randint(0, problem_size - 1)
    while(i == j):
        j = random.randint(0, problem_size - 1)

    sign = 1

    if random.random() < 0.5:
        sign = -1

    coefficient = random.randint(1,5)

    A2[i, :] += A[j, :] * sign * coefficient
    y2[i] += y[j] * sign * coefficient

    if np.max(np.abs(A2)) <= 2000000000 and np.max(np.abs(y2)) <= 2000000000:
        A[i, :] += A[j, :] * sign * coefficient
        y[i] += y[j] * sign * coefficient
    else:
        A2 = A.copy()
        y2 = y.copy()

    print(A)

print(answers.tolist())

with open(f"{problem_name}_{problem_size}.txt", 'w') as out:
    out.write("size:\n")
    out.write(f"{problem_size}")
    out.write("\nanswer:\n")
    out.write(','.join(str(x) for x in answers.tolist()))
    out.write("\n")
    out.write("A:\n")
    for i in range(problem_size):
        for j in range(problem_size):
            if A[i,j] != 0:
                out.write(f"{j},{A[i,j]};")
        out.write("\n")
    out.write("b:\n")
    out.write(','.join(str(x) for x in y.tolist()))
    out.write("\nend.\n")
