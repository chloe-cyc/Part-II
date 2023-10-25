#from scipy.optimize import minimize as min
from optimizer_find_windows import optimize
import numpy as np
import pandas as pd

column_names = ["Time"]
exc_data_path = "Data/mash_exc300K.dat"

# DATA IMPORT - Using Pandas because it is easier to me
for i in range(1,8):
    column_names.append(str(i))
exc_df = pd.read_csv(exc_data_path, delimiter=" ", names=column_names)
exc_df = exc_df[(exc_df.index % 10 == 0) | (exc_df.index == len(exc_df.index) - 1)]
exc_values = exc_df.values
eq_pop = exc_values[-1][1:8]
full_time = exc_df["Time"].values

#initial guess for kappa
no_states = 7#number of states
num_elements = no_states*(no_states-1)//2
initial_guess_kappa = np.full(num_elements,0.01) 


data = optimize(exc_values, initial_guess_kappa,eq_pop,full_time)[1]


np.savetxt(f"optimization_using_all_data.dat", data,delimiter="\t")

