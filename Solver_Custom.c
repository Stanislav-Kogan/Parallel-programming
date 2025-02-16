#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lu_decompose(double **A, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = k + 1; i < n; i++) {
            if (A[k][k] == 0.0) {
                fprintf(stderr, "Error: Zero pivot encountered at %d\n", k);
                exit(EXIT_FAILURE);
            }
            double factor = A[i][k] / A[k][k];
            A[i][k] = factor;
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= factor * A[k][j];
            }
        }
    }
}

void forward_substitute(double **A, double *b, double *y, int n) {
    for (int i = 0; i < n; i++) {
        y[i] = b[i];
        for (int j = 0; j < i; j++) {
            y[i] -= A[i][j] * y[j];
        }
    }
}

void backward_substitute(double **A, double *y, double *x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
}

int main() {
    int n = -1;
    int c;
    int c_prev = ' ';
    FILE *file;
    char *problem_name = "input";
    char problem_file[64]; strcpy(problem_file, problem_name);
    strcat(problem_file, ".txt");

    char problem_file_solved[64];strcpy(problem_file_solved, problem_name);
    strcat(problem_file_solved, "_solved_C.txt");

    file = fopen(problem_file, "r");

    int reading_state = 0;
    char str[64];
    int str_len = 0;
    int row_number = 0;
    int row_element = -1;

    double **A;
    double *b;
    double *y;
    double *x;

    if (file) {
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
                    n = atoi(str);
                    printf("\nproblem_size: %d\n", n);
                    str_len = 0;

                    A = malloc(n * sizeof(double *));
                    for (int i = 0; i < n; i++) {
                        A[i] = malloc(n * sizeof(double));
                        for (int j = 0; j < n; j++){
                            A[i][j] = 0;
                        }
                    }

                    b = malloc(n * sizeof(double));
                    y = malloc(n * sizeof(double));
                    x = malloc(n * sizeof(double));
                }
            }
            else if(reading_state == 2){
                if(c == '\n') reading_state = 3;
            }
            else if(reading_state == 3){
                if(c == '\n') reading_state = 4;
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
                        A[row_number][row_element++] = atof(str);
                    }
                    if(c == ',')
                    {
                        row_element = atoi(str);
                    }
                    if(c == '\n') {row_number++; printf("\n"); row_element=0;}
                    if(row_number == n) reading_state = 6;
                }
            }
            else if(reading_state == 6){
                if(c == '\n') reading_state = 7;
            }
            else if(reading_state == 7){
                if((c != ',') && (c != '\n')) str[str_len++] = c;
                else{
                    str[str_len++] = '\0';
                    printf("b[%d] = %s;\t", row_element, str);
                    b[row_element++] = atof(str);
                    str_len = 0;
                }
                if(c == '\n') {printf("\n"); reading_state = 8;}
            }
            c_prev = c;
        }
        fclose(file);
    } 

    lu_decompose(A, n);
    forward_substitute(A, b, y, n);
    backward_substitute(A, y, x, n);

    FILE *fp;
    fp = fopen(problem_file_solved, "w");
    if (fp == NULL) {
        printf("Error opening file!");
        return 1;
    }
    printf("Solution vector x:\n");
    for (int i = 0; i < n; i++) {
        printf("%.1f  ", x[i]);
        fprintf(fp, "%.1f", x[i]);
        if(i<n-1) fprintf(fp, ",");
    }
    fprintf(fp, "\n");
    fclose(fp);
        
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }

    return 0;


    free(A);
    free(b);
    free(y);
    free(x);

    return 0;
}