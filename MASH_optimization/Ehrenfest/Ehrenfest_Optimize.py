import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from optimizer_optimated import Optimizer
import pandas as pd

#IMPORT DATA
column_names = ["Time"]
ehrenfest_path = "/u/dem/kebl6911/Part-II/MASH_optimization/Ehrenfest/ehrenfest.dat"
site_path = "/u/dem/kebl6911/Part-II/MASH_optimization/Data/mash_site77K.dat"

for i in range(1,8):
    column_names.append(str(i))
ehrenfest_df = pd.read_csv(ehrenfest_path, delimiter=" ", names=column_names)
ehrenfest_df = ehrenfest_df[(ehrenfest_df.index % 10 == 0) | (ehrenfest_df.index == len(ehrenfest_df.index) - 1)]

# classical = "../Data/mash_site77K.dat"
# class_df = pd.read_csv(classical, delimiter=" ", names=column_names)
# class_df = class_df[(class_df.index % 10 == 0) | (class_df.index == len(site_df.index) - 1)]

# All the values within the df
ehr_eq = ehrenfest_df.values 
full_time = ehrenfest_df["Time"].values

# #Define ts
# t0 = 0
#t1 = 500
t2 = 10000
t3 = full_time[-1]

#Only consider values below t2
site_df = ehrenfest_df[ehrenfest_df["Time"]<=t2]
start_to_t2= site_df.values #DF to NP

no_states = 7
eq_pop = ehr_eq[-10]
eq_pop = np.mean(eq_pop, axis=0)

print(eq_pop)
eq_pop = eq_pop[1:8]

save_kappas = []
save_resid1 = [] # Full Train
save_resid2 = [] # Half Train
save_resid3 = [] # Extrapolation
save_t = []
save_p1s = []
save_neg = []

t1s = 20*np.arange(1,70)
#t1s = np.array([80])
for t1 in t1s:
    try:
        save_t.append(t1)
        t0 = t1/2
        print(t1)
        data = site_df[site_df["Time"]<=t1]
        
        start_to_t1= data.values #DF to NP
        t0_to_t1_full = start_to_t1[start_to_t1[:,0]>=t0]
        t0_to_t1_pop = t0_to_t1_full[:,1:]
        t0_to_t1_time = t0_to_t1_full[:,0]

        t1_to_t2_full = start_to_t2[start_to_t2[:,0]>=t1]
        t1_to_t2_time = t1_to_t2_full[:,0]
        t1_to_t2_pop = t1_to_t2_full[:,1:]

        #Array of t0 to use
        start_to_t1_time = start_to_t1[:,0]
        
        # Train using function+ Calculate residual for full training set relative to itself:
        optimize_init = Optimizer(start_to_t1, eq_pop, no_states, initial_kappas=None)
        rms, optimized_kappas= optimize_init.run()
        save_kappas.append(optimized_kappas)
        save_resid1.append(rms)

        #Calculate residual for full training set relative to 
        p0 = optimize_init.p0
        save_neg.append(optimize_init.time_size*7)
        # Calculate Residual for extrapolation set
        predict_pop = optimize_init.predict(p0,t1_to_t2_time,"foward",optimized_kappas)
        time_size = len(t1_to_t2_time)
        print(time_size)
        resid2 = t1_to_t2_pop-predict_pop
        resid2 = np.sqrt(np.sum(np.abs(resid2)**2)/(time_size*no_states))
        save_resid2.append(resid2)

        #Calculate Residual for Half training set
        predict_pop = optimize_init.predict(p0,t0_to_t1_time,"back",optimized_kappas)
        time_size = len(t0_to_t1_time)
        print(time_size)
        resid_half = t0_to_t1_pop - predict_pop
        resid3 = np.sqrt(np.sum(np.abs(resid_half)**2)/(time_size*no_states))
        save_resid3.append(resid3)

    except Exception as e:
        print("failed for t2=",t1)
        print(e)
        continue


#SAVE THE DATA
save_kappas = np.array(save_kappas)
save_resid1 = np.column_stack((save_t,save_resid1))
save_resid2 = np.column_stack((save_t,save_resid2))
save_resid3 = np.column_stack((save_t,save_resid3))
save_neg = np.column_stack((save_t,save_neg))

np.savetxt(f"scan_0_5000_site77_resid1.dat", save_resid1, delimiter = "\t")
np.savetxt(f"scan_0_5000_site77_resid2.dat", save_resid2, delimiter = "\t")
np.savetxt(f"scan_0_5000_site77_resid3.dat", save_resid3, delimiter = "\t")
np.savetxt(f"scan_0_5000_site77_neghalf.dat", save_neg, delimiter = "\t")
np.savetxt(f"scan_0_5000_site77_kappas.dat", save_kappas, delimiter = "\t")
