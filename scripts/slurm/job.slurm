#!/bin/bash
#SBATCH --job-name="matrix-completion-many"   # Sensible name for the job
#SBATCH --account=mikalst   # Account for consumed resources
#SBATCH --nodes=1             # Allocate 1 nodes for the job
#SBATCH --cpus-per-task=20
#SBATCH --time=00-00:10:00    # Upper time limit for the job (DD-HH:MM:SS)
#SBATCH --partition=WORKQ

module load intel/2018b
module load Python/3.6.6
module list

matlab -nodisplay -nodesktop -nosplash -nojvm -r "test"
