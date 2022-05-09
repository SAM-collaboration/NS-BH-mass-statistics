import numpy as np
import pandas as pd 
import glob, os
import scipy.stats as st

os.chdir("/Users/mmeskhi/Desktop/")

if __name__ == '__main__':
    for file in glob.glob("*.csv"):
        metrics = pd.read_csv(file)
        kl_values = metrics['KL']
        tv_values = metrics['TV']
        kl_ci = st.t.interval(0.95, len(kl_values)-1, loc=np.mean(kl_values), scale=st.sem(kl_values))
        tv_ci = st.t.interval(0.95, len(tv_values)-1, loc=np.mean(tv_values), scale=st.sem(tv_values))
        print(f"{file} mean and ci: {kl_values.mean():.4f}, {tv_values.mean():.4f}, {kl_ci}, {tv_ci}")