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

SN_FLAG = "black_holes"
ENGINE_1 = "../../../data/ail_pdr_meta/sneos_meta_reduced_clean.csv"
# ENGINE_2 = "../../../data/ail_pdr_meta/engine2/sneos_meta_e2.txt"
GRID_SIZE = 2**15
NUM_NOISE = 5000
BANDWITDH = 1.0

def main():
    output_kl, output_tv = pd.DataFrame(), pd.DataFrame()
    raw_data = preprocess.process_data(file_path=ENGINE_1)
    states, states_data = preprocess.split_states(raw_data)

    for state, state_data in enumerate(states_data):
        sample = utils.generate_sample(state_data, NUM_NOISE)
        grid = utils.generate_grid(sample, GRID_SIZE)

        if SN_FLAG == "black_holes":
            metrics = compute.compute_bh(sample, state, grid, SN_FLAG)
        elif SN_FLAG == "ns_isolated":
            metrics = compute.compute_nsi(sample, state, grid, SN_FLAG)
        elif SN_FLAG == "ns_double":
            metrics = compute.compute_nsd(sample, state, grid, SN_FLAG)
        else:
            print("Incorrect or no star flag passed.")
            sys.exit(0)




if __name__ == '__main__':
    main()
    