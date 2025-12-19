#!/bin/bash
#SBATCH -J mixed-parallel-sweep
#SBATCH -A m4999                     
#SBATCH -q regular                     
#SBATCH -C gpu                         
#SBATCH -t 02:00:00                    
#SBATCH -N 1                           
#SBATCH -G 4                           
#SBATCH --ntasks-per-node=1            

echo "Running on: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"

# activate virtual environment
source ~/.conda/envs/straggler_infer/bin/activate

# run your script
~/ccl-bench-straggler-scale-up/trace_gen/inference/run_experiment_sweep.sh