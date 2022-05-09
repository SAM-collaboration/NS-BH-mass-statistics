import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz

import densities
import utils
import compute
import preprocess
from metrics import kl_div, tv_dist

SN_FLAG = "ns_isolated"
GRID_SIZE = 2**15
ENGINE_1 = "../../../data/ail_pdr_meta/sneos_meta.txt"
ENGINE_2 = "../../../data/ail_pdr_meta/engine2/sneos_meta_e2.txt"
RESULTS = {SN_FLAG: {"ENGINE_1": {'KL': [], 'TV': []},
                     "ENGINE_2": {'KL': [], 'TV': []}}}

if __name__ == '__main__':
    for ENGINE, ENGINE_PATH in zip(["ENGINE_1", "ENGINE_2"], [ENGINE_1, ENGINE_2]):
        for i in range(100):
            raw_data = preprocess.process_data(file_path=ENGINE_PATH)
            data = pd.concat([raw_data]*5000, ignore_index=True)
            data = utils.compute_noise(data)
            data['weights'] = utils.compute_weights(data)
            states_data = preprocess.split_states(data)[:1]  # EOS LIMITER
            grid = utils.generate_grid(data, GRID_SIZE)

            if SN_FLAG == "black_holes":
                output_metrics = compute.compute_bh(states_data, grid, SN_FLAG)
                RESULTS[SN_FLAG][ENGINE]['KL'].append(output_metrics['KL'])
                RESULTS[SN_FLAG][ENGINE]['TV'].append(output_metrics['TV'])
            elif SN_FLAG == "ns_isolated":
                output_metrics = compute.compute_nsi(states_data, grid, SN_FLAG)
                RESULTS[SN_FLAG][ENGINE]['KL'].append(output_metrics['KL'])
                RESULTS[SN_FLAG][ENGINE]['TV'].append(output_metrics['TV'])
            elif SN_FLAG == "ns_double":
                output_metrics = compute.compute_nsd(states_data, grid, SN_FLAG)
                RESULTS[SN_FLAG][ENGINE]['KL'].append(output_metrics['KL'])
                RESULTS[SN_FLAG][ENGINE]['TV'].append(output_metrics['TV'])
            else:
                print("Incorrect or no star flag passed.")
                sys.exit(0)

    result_KL = list(itertools.chain.from_iterable(RESULTS[SN_FLAG]["ENGINE_1"]['KL']))
    result_TV = list(itertools.chain.from_iterable(RESULTS[SN_FLAG]["ENGINE_1"]['TV']))
    
    print(np.mean(result_KL), np.std(result_KL))
    print(np.mean(result_TV), np.std(result_TV))