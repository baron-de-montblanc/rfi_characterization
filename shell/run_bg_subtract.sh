#!/bin/bash

#SBATCH -J bg_subtract
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --account=jpober-condo

# For array jobs, limit the number of simultaneously running jobs using '%'
#SBATCH --array=0-228%30

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e /oscar/data/jpober/jmduchar/Research/mcgill25/rfi_characterization/shell/slurm_out/bg_subtract-%A-%a.out
#SBATCH -o /oscar/data/jpober/jmduchar/Research/mcgill25/rfi_characterization/shell/slurm_out/bg_subtract-%A-%a.out

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

source /gpfs/runtime/opt/anaconda/2022.05/etc/profile.d/conda.sh
conda activate mcgill25

python /users/jmduchar/data/jmduchar/Research/mcgill25/rfi_characterization/python/run_bg_subtract.py ${SLURM_ARRAY_TASK_ID}