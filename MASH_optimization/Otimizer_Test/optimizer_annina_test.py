from optimizer_annina import optimize
import pandas as pd
import time
import numpy as np

column_names = ["Time"]
exciton_data_path = "Data/mash_exc300K.dat"

for i in range(1,8):
    column_names.append(str(i))
exciton_df = pd.read_csv(exciton_data_path, delimiter=" ", names=column_names)
exciton_df = exciton_df[(exciton_df.index % 30 == 0) | (exciton_df.index == len(exciton_df.index) - 1)]
end_time_values = np.array(exciton_df['Time'].iloc[4::5])
exciton_values = exciton_df.values
eq_pop = exciton_values[-1][1:8]
full_time = exciton_df["Time"].values
#initial guess for kappa
no_states = 7#number of states
num_elements = no_states*(no_states-1)//2
initial_guess_kappa = np.full(num_elements,0.01)
residual_sum = []
t_max = end_time_values[-1]

optimized_eigenvalues_save = []
optimized_kappas_save = []

for index, end_time in enumerate(end_time_values, start=1):
    print(f"run{index}")
    exciton_data = exciton_values[exciton_values[:, 0] < end_time]
    optimized_data, optimized_eigenvalues, optimized_kappas = optimize(exciton_data,eq_pop,t_max,initial_guess_kappa, exciton_values, full_time)
    optimized_eigenvalues_save.append(optimized_eigenvalues)
    optimized_kappas_save.append(optimized_kappas)
    np.savetxt(f"Chloe_Test_File/Plotting_Windows_Annina/{end_time}_{index}.dat", optimized_data, delimiter="\t")

np.savetxt(f"Chloe_Test_File/eigenvals_Annina.dat", optimized_eigenvalues_save)
np.savetxt(f"Chloe_Test_File/kappa_Annina.dat", optimized_eigenvalues_save)

# residual_sum = []
# kappa_saved =[]