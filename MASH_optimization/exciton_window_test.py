from optimizer import optimize
import pandas as pd
import time
import numpy as np

column_names = ["Time"]
exciton_data_path = "/u/dem/kebl6911/Part-II/MASH_optimization/Data/mash_exc300K.dat"
for i in range(1,8):
    column_names.append(str(i))
exciton_df = pd.read_csv(exciton_data_path, delimiter=" ", names=column_names)
exciton_df = exciton_df[(exciton_df.index % 40 == 0) | (exciton_df.index == len(exciton_df.index) - 1)]
end_time_values = np.array(exciton_df['Time'].iloc[9::10])
np.delete(end_time_values, -2)
exciton_values = exciton_df.values
print(exciton_values)

eq_pop = exciton_values[-1][1:8]
print(eq_pop)
full_time = exciton_df["Time"].values


#initial guess for kappa
no_states = 7#number of states
num_elements = no_states*(no_states-1)//2
initial_guess_kappa = np.full(num_elements,0.01)
residual_sum = []

for index, end_time in enumerate(end_time_values, start=1):
    exciton_data = exciton_values[exciton_values[:, 0] < end_time]
    optimized_data,tmp,loss = optimize(exciton_data,initial_guess_kappa, eq_pop, full_time, exciton_values)
    np.savetxt(f"Window_data/{end_time}_{index}.dat", optimized_data, delimiter="\t")
    residual_sum.append(loss)
    print(f"run{index}")

np.savetxt(f"cost.dat", residual_sum, delimiter="\t")
# residual_sum = []
# kappa_saved =[]
