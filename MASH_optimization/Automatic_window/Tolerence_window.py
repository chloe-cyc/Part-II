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
site_df = site_df[site_df["Time"] <=1000]
site_values = site_df.values 
site_full = site_values[:,1:] #All the population data from t=0 to t=1000, exclusion of time intervals

#Extracting the equilibrium population and full length of time for this data
eq_pop = site_eq[-1][1:8]
full_time = site_df["Time"].values

#initial guess for kappa and p
no_states = 7#number of states
num_elements = no_states*(no_states-1)//2
prev_kap= np.full(num_elements,0.01) 


#Start time - cut data after this time
#start_time_values = np.array(site_df['Time'].iloc[4::5]) 
start_time_values = np.array(site_df['Time'])
start_time_values = start_time_values[2:]

tolerence_diff= 0.01

percent_diff_list = []
start_time_list = []
for index, start_time in enumerate(start_time_values, start=1):
    print(f"run {index}")
    
    #Only consider values that are smaller than the end time
    site_data = site_values[site_values[:,0] <= start_time]
    initial_guess_p = site_data[0,:]
    initial_guess_p = initial_guess_p[1:]
    percent_diff, optimized_data, prev_kap_1 = optimize(site_data, prev_kap, initial_guess_p, eq_pop, full_time, site_full)
    prev_kap = prev_kap_1
    percent_diff_list.append(percent_diff)
    start_time_list.append(start_time)
    np.savetxt(f"full_data/{start_time}_{index}.dat", optimized_data, delimiter="\t")

    if abs(percent_diff_list[index-2]-percent_diff_list[index-1]) <= tolerence_diff and index>=3: #Comparing to a fixed tolerence
    
        print("Percentage difference has met tolerence condition")
        break


start_time_list = np.array(start_time_list)
residual_data = np.column_stack((start_time_list,percent_diff_list))
np.savetxt(f"Percentage_diff.dat", residual_data,delimiter="\t")