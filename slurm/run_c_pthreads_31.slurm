#!/bin/bash

#SBATCH --job-name=compare_job
#SBATCH --output=compare_output_%j.log
#SBATCH --error=compare_error_%j.log
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks=31
#SBATCH --cpus-per-task=1

#Очистка загруженных модулей и загрузка нужных
module purge
module load compiler/gcc/9
module load mpi/openmpi/4.0.1/gcc/9

#Компиляция программ
gcc Solver_pthreads.c -o Solver_pthreads -lpthread

#Генерация данных и запуск
./Solver_pthreads 31      