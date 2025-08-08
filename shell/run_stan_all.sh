#!/bin/bash

#SBATCH -J run_stan_all
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=jpober-condo
#SBATCH --array=0-4

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e /oscar/data/jpober/jmduchar/Research/mcgill25/rfi_characterization/shell/slurm_out/run_stan_all-%A-%a.out
#SBATCH -o /oscar/data/jpober/jmduchar/Research/mcgill25/rfi_characterization/shell/slurm_out/run_stan_all-%A-%a.out

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate stan

python /users/jmduchar/data/jmduchar/Research/mcgill25/rfi_characterization/python/run_stan.py all p$SLURM_ARRAY_TASK_ID