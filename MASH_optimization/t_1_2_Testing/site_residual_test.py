#from scipy.optimize import minimize as min
from optimizer_find_windows import optimize
import numpy as np
import pandas as pd

column_names = ["Time"]
site_data_path = "/u/dem/kebl6911/Part-II/MASH_optimization/Data/mash_site300K.dat"

# DATA IMPORT - Using Pandas because it is easier to me
for i in range(1,8):
    column_names.append(str(i))
site_df = pd.read_csv(site_data_path, delimiter=" ", names=column_names)
site_df = site_df[(site_df.index % 10 == 0) | (site_df.index == len(site_df.index) - 1)]
site_eq = site_df.values 
full_time = site_df["Time"].values
#
site_df = site_df[site_df["Time"]<=1000]
site_values = site_df.values
site_full = site_values[:,1:]

#Extracting the equilibrium population and full length of time for this data
eq_pop = site_eq[-1][1:8]
#full_time = full_time[full_time<=3000]

#initial guess for kappa and p
no_states = 7#number of states
num_elements = no_states*(no_states-1)//2
prev_kap= np.full(num_elements,0.01) 

#Start time - cut data after this time
start_time_values = np.array(site_df['Time'])
t_one=800
# t_two = 1000
weight=2
start_time_values = start_time_values[start_time_values<=t_one]
start_time_values = start_time_values[:-10]
residual_sum = []
residual_sum_2 = []
final_pop = []
for index, start_time in enumerate(start_time_values, start=1):
    print(f"run {index}")
    #Only consider values that are smaller than the end time
    site_data = site_values[site_values[:,0] >= start_time]
    initial_guess_p = site_data[0,:]
    initial_guess_p = initial_guess_p[1:]
    residual_1, residual_2, optimized_data, prev_kap_1,calc_final = optimize(site_data, prev_kap, initial_guess_p, eq_pop, full_time,t_one,weight) #t_one,t_two, full_time)
    prev_kap = prev_kap_1
    residual_sum.append(residual_1)
    residual_sum_2.append(residual_2)
    final_pop.append(calc_final)
    #kappa_saved.append(index, least_squares_result)
    np.savetxt(f"/u/dem/kebl6911/Part-II/MASH_optimization/t_1_2_Testing/w2_t1_800_t2_1000/{start_time}_{index}.dat", optimized_data, delimiter="\t")

residual_data1 = np.column_stack((start_time_values,residual_sum))
residual_data2 = np.column_stack((start_time_values,residual_sum_2))
final_pop = np.column_stack((start_time_values,final_pop))
np.savetxt(f"/u/dem/kebl6911/Part-II/MASH_optimization/t_1_2_Testing/w2_800_1000_resid1.dat",residual_data1, delimiter = "\t")
np.savetxt(f"/u/dem/kebl6911/Part-II/MASH_optimization/t_1_2_Testing/w2_800_1000_resid2.dat",residual_data2, delimiter = "\t")
np.savetxt(f"/u/dem/kebl6911/Part-II/MASH_optimization/t_1_2_Testing/w_800_1000_final_pop.dat", final_pop,delimiter="\t")