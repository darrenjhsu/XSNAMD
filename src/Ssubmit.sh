#!/bin/bash

#SBATCH --job-name CUDA_test
#SBATCH --partition=PASCAL
#SBATCH --qos normal
#SBATCH --nodes 1
#SBATCH --cpus-per-task 12
#SBATCH --gres=gpu:1
#SBATCH --time 01:20:00
#SBATCH --output CUDA_test.out
#SBATCH --mem=32768

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/djh992/lib/gsl/lib
module load cuda/9.1.85
date +%s%3N
#for i in {1..100}; do
#srun --gres=gpu:1 tclsh test.tcl
#done
#srun --gres=gpu:1 nvprof --analysis-metrics -o speedtest25.txt ./a.out 
#srun --gres=gpu:1 nvprof ./structure_calc
#srun --gres=gpu:1 ./structure_calc
#srun --gres=gpu:1 cuda-memcheck ./structure_calc 
#srun --gres=gpu:1 cuda-memcheck ./fit_initial 
#srun --gres=gpu:1 ./a.out
#srun --gres=gpu:1 cuda-memcheck ./a.out
#cuda-memcheck ./a.out
#srun --gres=gpu:1 nvprof ./a.out
./fit_initial.out
date +%s%3N
