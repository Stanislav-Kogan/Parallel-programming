import numpy as np
from mpi4py import MPI
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    answer = None

    if rank == 0:
        with open(f"input.txt", "r") as inp:
            inp.readline()
            N = (int)(inp.readline())
            print(f"Problem_size: {N}")
            inp.readline()
            str_arr = inp.readline().replace('\n', '').split(',')
            np_arr = np.array(str_arr).astype(float)
            answer = np_arr[:, np.newaxis]
            inp.readline()
            A = np.zeros((N,N))
            for i in range(N):
                str_arr = inp.readline().replace('\n', '').split(';')
                for part in str_arr:
                    if part != "":
                        parts = part.split(',')
                        #print(int(parts[0]))
                        A[i, int(parts[0])] = float(parts[1])
            inp.readline()
            str_arr = inp.readline().replace('\n', '').split(',')
            np_arr = np.array(str_arr).astype(float)
            b = np_arr[:, np.newaxis]
    else:
        A, b, N = None, None, None

    N = comm.bcast(N, root=0)

    before_time  = time.time()

    if rank == 0:
        base_chunk = N // size
        remainder = N % size
        chunk_sizes = [base_chunk + 1 if i < remainder else base_chunk for i in range(size)]
        counts = [cs * N for cs in chunk_sizes]
        displacements = [sum(counts[:i]) for i in range(size)]
        A_flat = A.flatten()
    else:
        chunk_sizes, counts, displacements = None, None, None
        A_flat = None

    chunk_sizes = comm.bcast(chunk_sizes, root=0)
    chunk_size = chunk_sizes[rank]

    start_rows = [sum(chunk_sizes[:i]) for i in range(size)]
    my_start = start_rows[rank]

    local_A = np.empty((chunk_size, N), dtype=np.float64)

    comm.Scatterv([A_flat, counts, displacements, MPI.DOUBLE], local_A, root=0)

    for k in range(N):
        owner = 0
        while owner < size - 1 and k >= start_rows[owner + 1]:
            owner += 1

        if rank == owner:
            local_k = k - start_rows[owner]
            pivot_row = np.copy(local_A[local_k])
        else:
            pivot_row = np.empty(N, dtype=np.float64)

        comm.Bcast(pivot_row, root=owner)

        for i in range(chunk_size):
            global_row = my_start + i
            if global_row > k:
                multiplier = local_A[i, k] / pivot_row[k]
                local_A[i, k] = multiplier
                local_A[i, k+1:] -= multiplier * pivot_row[k+1:]

    local_flat = local_A.flatten()
    if rank == 0:
        A_decomposed = np.empty(N * N, dtype=np.float64)
    else:
        A_decomposed = None

    comm.Gatherv(local_flat, [A_decomposed, counts, displacements, MPI.DOUBLE], root=0)

    if rank == 0:
        A_decomposed = A_decomposed.reshape(N, N)
        L = np.tril(A_decomposed, -1) + np.eye(N)
        U = np.triu(A_decomposed)

        y = np.zeros(N)
        for i in range(N):
            y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]

        x = np.zeros(N)
        for i in reversed(range(N)):
            x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

        after_time = time.time()

        print(f'Solved in {(after_time - before_time) * 1000.} ms')

        print(f"Answer:\t\t{'  '.join(str(s[0]) for s in answer.tolist())}")

        print(f"Solution:\t{'  '.join(str(s) for s in x.tolist())}")

        print(f"\nSUCCESS")

        with open(f"output_python_mpi.txt", 'w') as out:
            out.write(f'CALCULATION TIME {(after_time - before_time) * 1000.} ms\n')
            out.write(','.join(str(s) for s in x.tolist()))
            out.write("\n")

if __name__ == "__main__":
    main()