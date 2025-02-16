#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

int MAX_THREADS = 4; 

typedef struct {
    double **A;
    int n;
    int k;
} ThreadData;

void *lu_decomposition_thread(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    double **A = data->A;
    int n = data->n;
    int k = data->k;

    for (int i = k + 1; i < n; i++) {
        A[i][k] /= A[k][k];
        for (int j = k + 1; j < n; j++) {
            A[i][j] -= A[i][k] * A[k][j];
        }
    }
    pthread_exit(NULL);
}

void forward_substitution(double **L, double *b, double *y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= L[i][j] * y[j];
        }
    }
}

void backward_substitution(double **U, double *y, double *x, int n) {
    double *x1 = (double*)malloc(n * sizeof(double));
    for (int i = n - 1; i >= 0; i--) {
        x1[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x1[i] -= U[i][j] * x1[j];
        }
        x1[i] /= U[i][i];
    }
}

double** allocate_matrix(int n) {
    double **matrix = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double *)malloc(n * sizeof(double));
    }
    return matrix;
}

void free_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void lu_decomposition(double **A, int n) {
    pthread_t threads[MAX_THREADS];
    ThreadData threadData[MAX_THREADS];

    for (int k = 0; k < n; k++) {
        int num_threads = (n - k - 1 < MAX_THREADS) ? (n - k - 1) : MAX_THREADS;
        for (int t = 0; t < num_threads; t++) {
            threadData[t].A = A;
            threadData[t].n = n;
            threadData[t].k = k;
            pthread_create(&threads[t], NULL, lu_decomposition_thread, &threadData[t]);
        }
        for (int t = 0; t < num_threads; t++) {
            pthread_join(threads[t], NULL);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        MAX_THREADS = atoi(argv[1]); 
    }
    //printf("Using %d threads\n", MAX_THREADS);
    
    int N; 
    double** global_A = NULL;
    double* global_b = NULL;
    double* global_a = NULL;
    double* local_A = NULL;
    int* counts = NULL;
    int* displs = NULL;
    int* P = NULL;
    double* x;
    double* x1;
    double* y;

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
                //printf("\nproblem_size: %d\n", N);
                str_len = 0;

                global_A = malloc(N * sizeof(double *));
                for (int i = 0; i < N; i++) {
                    global_A[i] = malloc(N * sizeof(double));
                    for(int j = 0; j < N; j++){
                        global_A[i][j] = 0;
                    }
                }
                global_b =  (double*)malloc(N * sizeof(double));
                global_a =  (double*)malloc(N * sizeof(double));
                x = (double*)malloc(N * sizeof(double));
                x1 = (double*)malloc(N * sizeof(double));
                y = (double*)malloc(N * sizeof(double));
            }
        }
        else if(reading_state == 2){
            if(c == '\n') reading_state = 3;
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
                //printf(str);
                str[str_len] = '\0';
                str_len = 0;
                if(c == ';'){
                    //printf("A[%d][%d] = %s;\t", row_number, row_element, str);
                    global_A[row_number][row_element++] = atof(str);
                }
                if(c == ',')
                {
                    row_element = atoi(str);
                }
                if(c == '\n') row_number++;
                if(row_number == N) {reading_state = 6; row_element = 0;}
            }
        }
        else if(reading_state == 6){
            if(c == '\n') reading_state = 7;
        }
        else if(reading_state == 7){
            if((c != ',') && (c != '\n')) str[str_len++] = c;
            else{
                str[str_len++] = '\0';
                //printf("b[%d] = %s;\t", row_element, str);
                global_b[row_element++] = atof(str);
                str_len = 0;
            }
            if(c == '\n') {//printf("\n"); 
            reading_state = 8;}
        }
        c_prev = c;
    }
    fclose(file);
    
    clock_t begin = clock();

    lu_decomposition(global_A, N);
    forward_substitution(global_A, global_b, y, N);
    backward_substitution(global_A, y, x1, N);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;

    printf("\nCALCULATION TIME %lf ms\n", time_spent);

    FILE *fp;
    fp = fopen("output_C_pthreads.txt", "w");
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

    for (int i = 0; i < N; i++) {
        free(global_A[i]);
    }
    free(global_A);

    return 0;
}