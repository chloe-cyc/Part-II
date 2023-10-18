#from scipy.optimize import minimize as min
from optimizer_find_windows import optimize
import numpy as np
import pandas as pd
import time

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
residual_sum = []


#Start time
# end_time_values = np.array(exc_df['Time'].iloc[4::5])
end_time_values = np.array(exc_df['Time'])
end_time_values = end_time_values[2:]
end_time_values = end_time_values[end_time_values<=1000]


for index, end_time in enumerate(end_time_values, start=1):
    print(f"run {index}")
    #Only consider values that are larger than the start_time
    exc_data = exc_values[exc_values[:,0] <= end_time]
    residual_sum_val = optimize(exc_data, initial_guess_kappa,eq_pop,full_time)
    
    residual_sum.append(residual_sum_val)

    #kappa_saved.append(index, least_squares_result)
    #np.savetxt(f"Exc_residuals/limited_population/{end_time}_{index}.dat", optimized_data, delimiter="\t")


# residual_data = np.column_stack((end_time_values,residual_sum))
# np.savetxt(f"Exc_residuals/limited_population/residualdivided_data.dat", residual_data,delimiter="\t")


