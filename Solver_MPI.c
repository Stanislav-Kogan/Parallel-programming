#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

void print_matrix(double* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.4f ", A[i*N + j]);
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N; 
    double* global_A = NULL;
    double* global_b = NULL;
    double* global_a = NULL;
    double* local_A = NULL;
    int* counts = NULL;
    int* displs = NULL;
    int* P = NULL;
    double* x;

   
    FILE *file;

    file = fopen("input.txt", "r");
    int c;
    int c_prev = ' ';
    int reading_state = 0;
    char str[64];
    int str_len = 0;
    int row_number = 0;
    int row_element = 0;
    
    while ((c = getc(file)) != EOF)
    {
        if(reading_state == 0){
            if(c == '\n') reading_state = 1;
        }
        else if(reading_state == 1){
            if(c != '\n') str[str_len++] = c;
            else {
                str[str_len++] = '\0';
                reading_state = 2;
                row_element = 0;
                N = atoi(str);
                
                if(rank != 0) break;
                printf("\nproblem_size: %d\n", N);

                global_A = (double*)malloc(N * N * sizeof(double));
                global_b = (double*)malloc(N * sizeof(double));
                global_a = (double*)malloc(N * sizeof(double));
                x = (double*)malloc(N * sizeof(double));
                for (int i = 0; i < N; i++){
                    for (int j = 0; j < N; j++){
                        global_A[i*N+j] = 0;
                    }
                }
                
                if (global_A == NULL || global_b == NULL) {
                    fprintf(stderr, "Memory allocation failed on root!\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                str_len = 0;
            }
        }
        else if(reading_state == 2){
            if(c == '\n') {
                reading_state = 3;
            }
        }
        else if(reading_state == 3){
            if((c != ',') && (c != '\n')) str[str_len++] = c;
            else{
                str[str_len++] = '\0';
                //printf("b[%d] = %s;\t", row_element, str);
                global_a[row_element] = atof(str);
                x[row_element] = global_a[row_element];
                row_element++;

                str_len = 0;
            }
            if(c == '\n') {//printf("\n"); 
            row_element = 0; reading_state = 4;}
        }
        else if(reading_state == 4){
            if(c == '\n') reading_state = 5;
        }
        else if(reading_state == 5){
            if((c != ';') && (c != ',') && (c != '\n')) str[str_len++] = c;
            else{
                
                str[str_len] = '\0';
                str_len = 0;
                if(c == ';'){
                    //printf("A[%d][%d] = %s;\t", row_number, row_element, str);
                    
                    global_A[row_number*N+row_element] = atof(str);
                }
                else if(c == ',')
                {
                    row_element = atoi(str);
                }
                else if(c == '\n') row_number++; 
                if(row_number == N) {reading_state = 6; row_element = 0;}
            }
        }
        else if(reading_state == 6){
            if(c == '\n') reading_state = 7;
        }
        else if(reading_state == 7){
            if((c != ',') && (c != '\n')) str[str_len++] = c;
            else{
                str[str_len] = '\0';
                //printf("b[%d] = %s;\t", row_element, str);
                global_b[row_element++] = atof(str);
                str_len = 0;
            }
            if(c == '\n') { //printf("\n"); 
            reading_state = 8;}
        }
        c_prev = c;
    }
    fclose(file);

    MPI_Barrier(MPI_COMM_WORLD);
    clock_t begin = clock();

    counts = (int*)malloc(size * sizeof(int));
    displs = (int*)malloc(size * sizeof(int));

    if (counts == NULL || displs == NULL) {
        fprintf(stderr, "Memory allocation failed for counts/displs!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    int remainder = N % size;
    int sum = 0;
    for (int i = 0; i < size; i++) {
        counts[i] = (N / size + (i < remainder ? 1 : 0)) * N;
        displs[i] = sum;
        sum += counts[i];
    }
    int local_rows = counts[rank] / N; 

    local_A = (double*)malloc(local_rows * N * sizeof(double));
    if (local_A == NULL) {
        fprintf(stderr, "Memory allocation failed for local_A on rank %d!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Scatterv(global_A, counts, displs, MPI_DOUBLE, 
                 local_A, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    P = (int*)malloc(N * sizeof(int));
    if (P == NULL) {
        fprintf(stderr, "Memory allocation failed for P on rank %d!\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < N; i++) P[i] = i;

    for (int r; r < N; r++) {
        int k = 0;
        double local_max = -1.0;
        int local_max_row = -1;
        for (int i = 0; i < local_rows; i++) {
            int global_row = displs[rank] + i;
            if (global_row >= k) {
                double val = fabs(local_A[i*N + k]);
                if (val > local_max) {
                    local_max = val;
                    local_max_row = global_row;
                }
            }
        }

        struct { double val; int idx; } in, out;
        in.val = local_max;
        in.idx = local_max_row;
        MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        int pivot_row = out.idx;

        int owner_k = -1, owner_pivot = -1;
        for (int i = 0; i < size; i++) {
            if (displs[i] <= k && k < displs[i] + counts[i]) owner_k = i;
            if (displs[i] <= pivot_row && pivot_row < displs[i] + counts[i]) owner_pivot = i;
        }

        if (owner_k != owner_pivot) {
            
            double* send_buf = (double*)malloc(N * sizeof(double));
            double* recv_buf = (double*)malloc(N * sizeof(double));
            
            if (rank == owner_k) {
                int local_k = k - displs[rank];
                memcpy(send_buf, &local_A[local_k*N], N*sizeof(double));
                MPI_Sendrecv(send_buf, N, MPI_DOUBLE, owner_pivot, 0,
                            recv_buf, N, MPI_DOUBLE, owner_pivot, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                memcpy(&local_A[local_k*N], recv_buf, N*sizeof(double));
            }
            else if (rank == owner_pivot) {
                int local_pivot = pivot_row - displs[rank];
                memcpy(send_buf, &local_A[local_pivot*N], N*sizeof(double));
                MPI_Sendrecv(send_buf, N, MPI_DOUBLE, owner_k, 0,
                            recv_buf, N, MPI_DOUBLE, owner_k, 0,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                memcpy(&local_A[local_pivot*N], recv_buf, N*sizeof(double));
            }
            
            free(send_buf);
            free(recv_buf);
        }

        int temp = P[k];
        P[k] = P[pivot_row];
        P[pivot_row] = temp;

        double* pivot_row_data = (double*)malloc(N * sizeof(double));
        if (rank == owner_k) {
            int local_k = k - displs[rank];
            memcpy(pivot_row_data, &local_A[local_k*N], N*sizeof(double));
        }
        MPI_Bcast(pivot_row_data, N, MPI_DOUBLE, owner_k, MPI_COMM_WORLD);

        for (int i = 0; i < local_rows; i++) {
            int global_i = displs[rank] + i;
            if (global_i > k) {
                double factor = local_A[i*N + k] / pivot_row_data[k];
                local_A[i*N + k] = factor;
                for (int j = k+1; j < N; j++) {
                    local_A[i*N + j] -= factor * pivot_row_data[j];
                }
            }
        }
        
        free(pivot_row_data);
    }


    MPI_Gatherv(local_A, local_rows * N, MPI_DOUBLE, 
                global_A, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double* Pb = (double*)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) Pb[i] = global_b[P[i]];

        double* y = (double*)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            y[i] = Pb[i];
            for (int j = 0; j < i; j++) {
                y[i] -= global_A[i*N + j] * y[j];
            }
        }

        double* x1 = (double*)malloc(N * sizeof(double));
        for (int i = N-1; i >= 0; i--) {
            x1[i] = y[i];
            for (int j = i+1; j < N; j++) {
                x1[i] -= global_A[i*N + j] * x1[j];
            }
            x1[i] /= global_A[i*N + i];
        }

        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;

        printf("CALCULATION TIME %lf ms\n", time_spent);

        FILE *fp;
        fp = fopen("output_C_mpi.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!");
            return 1;
        }
        printf("Answer vector:\n");
        for (int i = 0; i < N; i++) {
            printf("%.1f  ", global_a[i]);
        }
        printf("\nSolution vector x:\n");
        fprintf(fp, "CALCULATION TIME %lf ms\n", time_spent);
        for (int i = 0; i < N; i++) {
            printf("%.1f  ", x[i]);
            fprintf(fp, "%.1f", x[i]);
            if(i<N-1) fprintf(fp, ",");
        }
        printf("\n\nSUCCESS\n");

        fprintf(fp, "\n");
        fclose(fp);

        free(Pb);
        free(y);
        free(x);
    }

    free(local_A);
    free(counts);
    free(displs);
    free(P);
    if (rank == 0) {
        free(global_A);
        free(global_b);
    }

    MPI_Finalize();
    return 0;
}
