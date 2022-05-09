import sys
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

import compute
import densities
import utils
import preprocess
from metrics import kl_div, tv_dist

SN_FLAG = "nss"
ENGINE_1 = "../../../../data/ail_pdr_meta/sneos_meta_reduced_clean_cold.csv"
GRID_SIZE = 2**15
NUM_BOOTSTRAPS = 2000
NUM_NOISE = 2000
BANDWITDH = 0.1


if __name__ == '__main__':
    output_kl, output_tv = pd.DataFrame(), pd.DataFrame()
    raw_data = preprocess.process_data(file_path=ENGINE_1)
    states, states_data = preprocess.split_states(raw_data)

    for state, state_data in zip(states, states_data):
        results = pd.DataFrame(columns=['KL', 'TV'])
        for boot_idx in range(NUM_BOOTSTRAPS):
            bootstrap_sample = utils.generate_bootstrap(state_data.copy(), NUM_NOISE)
            grid = utils.generate_grid(bootstrap_sample, GRID_SIZE)
            
            if SN_FLAG == "nss":
                metrics = compute.compute_nss(bootstrap_sample, state, grid, SN_FLAG)
                results.loc[len(results)] = metrics
            else:
                print("Incorrect or no star flag passed.")
                sys.exit(0)

        bootstrap_KL = results['KL'].to_numpy()
        bootstrap_TV = results['TV'].to_numpy()

        # CI_KL = st.norm.interval(alpha=0.9, loc=np.mean(bootstrap_KL), scale=st.sem(bootstrap_KL))
        # CI_TV = st.norm.interval(alpha=0.9, loc=np.mean(bootstrap_TV), scale=st.sem(bootstrap_TV)) 

        results.loc[len(results)] = {'KL': np.quantile(results['KL'].to_numpy(), [0.025, 0.975]),
                                     'TV': np.quantile(results['TV'].to_numpy(), [0.025, 0.975])}
        # results.loc[len(results)] = {'KL': CI_KL, 'TV': CI_TV} 

        results.loc[len(results)+1] = {'KL': bootstrap_KL.mean(), 'TV': bootstrap_TV.mean()}
        results.to_csv(f"./results_nss/{SN_FLAG}_{state}.csv")



