#from scipy.optimize import minimize as min
from optimizer_find_windows import optimize
import numpy as np
import pandas as pd
import time

column_names = ["Time"]
site_data_path = "Data/mash_site300K.dat"

# DATA IMPORT - Using Pandas because it is easier to me
for i in range(1,8):
    column_names.append(str(i))
site_df = pd.read_csv(site_data_path, delimiter=" ", names=column_names)
site_df = site_df[(site_df.index % 30 == 0) | (site_df.index == len(site_df.index) - 1)]
site_df = site_df[site_df.iloc[:,0] >=420.8860085880096]
site_values = site_df.values

eq_pop = site_values[-1][1:8]
full_time = site_df["Time"].values
full_time = full_time[full_time >=420.8860085880096]
print(full_time)

#initial guess for kappa
no_states = 7#number of states
num_elements = no_states*(no_states-1)//2
initial_guess_kappa = np.full(num_elements,0.01) 
residual_sum = []

#Start time

end_time_values = np.array(site_df['Time'].iloc[4::5])
end_time_values =end_time_values[:-2]


#Optimizing w/ initial guess for k
no_states = 7 #number of states
num_elements = no_states*(no_states-1)//2
initial_guess_kappa = np.full(num_elements,0.01)

residual_sum = []


for index, end_time in enumerate(end_time_values, start=1):
    print(f"run {index}")
    #Only consider values that are larger than the start_time
    site_data = site_values[site_values[:,0] <= end_time]
    residual, optimized_data = optimize(site_data, initial_guess_kappa,eq_pop,full_time,site_values)
    # residual_sum.append(residual)
    #kappa_saved.append(index, least_squares_result)
    np.savetxt(f"Site_data_windows/{end_time}_{index}.dat", optimized_data, delimiter="\t")

# residual_data = np.column_stack((start_time_values,residual_sum))
# np.savetxt(f"Site_dat_no_cut/residual_data.dat", residual_data,delimiter="\t")
