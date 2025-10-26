#!/bin/bash

#SBATCH -J run_stan
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --account=jpober-condo

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e /oscar/data/jpober/jmduchar/Research/mcgill25/rfi_characterization/shell/slurm_out/run_stan-%J.out
#SBATCH -o /oscar/data/jpober/jmduchar/Research/mcgill25/rfi_characterization/shell/slurm_out/run_stan-%J.out

source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate stan

python /users/jmduchar/data/jmduchar/Research/mcgill25/rfi_characterization/python/run_stan.py p0