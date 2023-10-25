#from scipy.optimize import minimize as min
from optimizer_find_windows import optimize
import numpy as np
import pandas as pd
import time

column_names = ["Time"]
exc_data_path = "/u/dem/kebl6911/Part-II/MASH_optimization/Data/mash_exc300K.dat"

# DATA IMPORT - Using Pandas because it is easier to me
for i in range(1,8):
    column_names.append(str(i))
exc_df = pd.read_csv(exc_data_path, delimiter=" ", names=column_names)
exc_df = exc_df[(exc_df.index % 20 == 0) | (exc_df.index == len(exc_df.index) - 1)]
exc_eq = exc_df.values
exc_df = exc_df[exc_df["Time"] <=1000]
exc_values = exc_df.values
exc_full = exc_values[:,1:]

#Extracting the equilibrium population and full length of time for this data
eq_pop = exc_eq[-1][1:8]
full_time = exc_df["Time"].values

#initial guess for kappa and p
no_states = 7#number of states
num_elements = no_states*(no_states-1)//2
prev_kappa = np.full(num_elements,0.01) 


#Start time - cut data after this time
#start_time_values = np.array(exc_df['Time'].iloc[4::5]) 
start_time_values = np.array(exc_df['Time'])
start_time_values = start_time_values[2:]


residual_sum = []

for index, start_time in enumerate(start_time_values, start=1):
    print(f"run {index}")
    #Only consider values that are smaller than the end time
    exc_data = exc_values[exc_values[:,0] >= start_time]
    initial_guess_p = exc_data[0,:]
    initial_guess_p = initial_guess_p[1:]
    residual_sum_val, optimized_data, kappa_optimized = optimize(exc_data, prev_kappa, initial_guess_p, eq_pop)
    prev_kappa=kappa_optimized
    residual_sum.append(residual_sum_val)

    #kappa_saved.append(index, least_squares_result)
    np.savetxt(f"exc_end_divfull/{start_time}_{index}.dat", optimized_data, delimiter="\t")

residual_data = np.column_stack((start_time_values,residual_sum))
np.savetxt(f"Residual_exc_end_divfull.dat", residual_data,delimiter="\t")

# residual_sum.append(residual_sum_val)

# #kappa_saved.append(index, least_squares_result)
# np.savetxt(f"exc_start_divwindow/{start_time}_{index}.dat", optimized_data, delimiter="\t")
