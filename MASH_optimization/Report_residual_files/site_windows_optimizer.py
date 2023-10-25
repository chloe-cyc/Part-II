#from scipy.optimize import minimize as min
from optimizer_find_windows import optimize
from sklearn.metrics import mean_squared_error
from scipy.linalg import norm
from scipy.optimize import minimize
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


# for index, end_time in enumerate(end_time_values, start=1):
#     print(f"run {index}")
#     #Only consider values that are larger than the start_time
#     exc_data = exc_values[exc_values[:,0] <= end_time]
#     residual_sum_val, optimized_data = optimize(exc_data, initial_guess_kappa,eq_pop,full_time,exc_values)
#     residual_sum.append(residual_sum_val)

#     #kappa_saved.append(index, least_squares_result)
#     np.savetxt(f"Exc_residuals/Divided_residuals/{end_time}_{index}.dat", optimized_data, delimiter="\t")


#Second optimizeation
#residualsum_perdata = optimize(data, initial_guess_kappa, eq_pop, full_time, exc_values)

tolerence = 0.19
residual_sum_data = []
current_data = np.empty((0, exc_values.shape[1]))
current_data = np.vstack((current_data, exc_values[0]))

# Iterate through the full dataset to add data points
for index, val in enumerate(exc_values, start=1):
    print(index)
    current_data = np.vstack((current_data, exc_values[index]))

    # Calculate optimization quality
    residual_sum_per_dat = optimize(current_data, initial_guess_kappa, eq_pop, full_time, exc_values)
    residual_sum_data.append(residual_sum_per_dat)

    # Check if the optimization quality is within tolerance
    if residual_sum_per_dat <= tolerence:
        break

# Access the last data point to determine the smallest acceptable
smallest_acceptable = current_data[-1, 0]
print(smallest_acceptable)