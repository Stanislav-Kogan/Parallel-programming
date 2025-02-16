#!/bin/bash

#SBATCH --job-name=compare_job
#SBATCH --output=compare_output_%j.log
#SBATCH --error=compare_error_%j.log
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

#Очистка загруженных модулей и загрузка нужных
module purge
module load python/3.9
module load mpi/openmpi/4.0.1/gcc/9

#Генерация данных и запуск
mpirun -np 4 python3 Solver_mpi.py         