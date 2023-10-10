from optimizer import optimize
import pandas as pd
import time
import numpy as np

start_time = time.time()

# DATA IMPORT - Using Pandas because it is easier to me
column_names = ["Time"]
site_data_path = "/u/dem/kebl6911/Part-II/MASH_optimization/Data/mash_site300K.dat"
for i in range(1,8):
    column_names.append(str(i))

site_df = pd.read_csv(site_data_path, delimiter=" ", names=column_names)
site_df =site_df[site_df["Time"]>=4.5e+02] #new df to retain old one
site_df = site_df[(site_df.index % 30 == 0) | (site_df.index == len(site_df.index) - 1)]

site_values = site_df.values()

no_states = 7#number of states
num_elements = no_states*(no_states-1)//2
initial_guess_kappa = np.full(num_elements,0.01)

optimize(site_values, initial_guess_kappa)[1]