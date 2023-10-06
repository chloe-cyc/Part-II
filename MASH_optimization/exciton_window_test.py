from optimizer import optimize
import pandas as pd
import time
import numpy as np

column_names = ["Time"]
exciton_data_path = "/u/dem/kebl6911/Part-II/MASH_optimization/Data/mash_site300K.dat"
for i in range(1,8):
    column_names.append(str(i))
exciton_df = pd.read_csv(exciton_data_path, delimiter=" ", names=column_names)
exciton_df = exciton_df[(exciton_df.index % 30 == 0) | (exciton_df.index == len(exciton_df.index) - 1)]
end_time_values = np.array(exciton_df['Time'].iloc[4::5])
exciton_values = exciton_df.values

#initial guess for kappa
no_states = 7#number of states
num_elements = no_states*(no_states-1)//2
initial_guess_kappa = np.full(num_elements,0.01)
residual_sum = []

for index, end_time in enumerate(end_time_values, start=1):
    exciton_data = exciton_values[exciton_values[:, 0] < end_time]
    optimized_data = optimize(exciton_data,initial_guess_kappa)[0]
    np.savetxt(f"Window_data/{end_time}_{index}.dat", optimized_data, delimiter="\t")
    print(f"run{index}")

# residual_sum = []
# kappa_saved =[]
