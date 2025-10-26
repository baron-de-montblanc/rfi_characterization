import sys
import numpy as np
import yaml
import matplotlib.pyplot as plt
import time

sys.path.append('/users/jmduchar/data/jmduchar/Research/mcgill25/rfi_characterization/python/')
from mapper import bg_subtract
from utils import get_ref_obsids, get_night

plt.style.use('seaborn-v0_8')
ABS_DIR = '/users/jmduchar/data/jmduchar/Research/mcgill25/'

def run_background_subtract(obs_idx):

    dirpath = ABS_DIR+'ssins_data/'
    with open(dirpath+'gridpoint_dict.yaml', 'r') as file:
        data = yaml.safe_load(file)

    ref_obsids = get_ref_obsids(data)
    pointing, obsids = get_night(data=data, target_obsid=ref_obsids[obs_idx])

    print(f"Running background subtraction on the night of {ref_obsids[obs_idx][:6]}" 
          +f" at pointing {pointing}...")
    subtracted_data = bg_subtract(data_dir=dirpath+'tars/', obsids=obsids, N_terms=24)[0]
    subtracted_data = subtracted_data[~np.isnan(subtracted_data)]
    
    print("Background subtraction complete! Saving to disk.")

    np.save(ABS_DIR+f"rfi_characterization/data_private/subtracted_data_{ref_obsids[obs_idx][:6]}_p{pointing}.npy",
            subtracted_data)


if __name__ == "__main__":
    
    t0 = time.time()
    
    if len(sys.argv) != 2:
        print("Usage: python run_background_subtract.py <obs_idx>")
        sys.exit(1)
    
    try:
        obs_idx = int(sys.argv[1])
    except ValueError:
        print("Error: obs_idx must be an integer.")
        sys.exit(1)

    run_background_subtract(obs_idx)
    print("Time elapsed:", time.time()-t0, "seconds.")
