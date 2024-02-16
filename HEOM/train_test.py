import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from optimizer_optimated import Optimizer
import pandas as pd

#IMPORT DATA

column_names = ["Time"]
data_path = "/u/dem/kebl6911/Part-II/HEOM/pop_site_tau-50_T-300_init-0_K_2_L_2.dat"

for i in range(1,8):
    column_names.append(str(i))
print(column_names)
df = pd.read_csv(data_path, delimiter=" ", names=column_names, usecols=range(len(column_names)))

df = df[(df.index % 10 == 0) | (df.index == len(df.index) - 1)]

# All the values within the df
values = df.values 
full_time = df["Time"].values

# #Define ts
# t0 = 0
#t1 = 500
t2 = 5000
t3 = full_time[-1]

#Only consider values below t2
site_df = df[df["Time"]<=t2]
start_to_t2= site_df.values #DF to NP

no_states = 7
eq_pop = values[-1]
eq_pop = eq_pop[1:8]

save_kappas = []
save_resid1 = [] # Full Train
save_resid2 = [] # Half Train
save_resid3 = [] # Extrapolation
save_t = []
save_p1s = []
save_neg = []

t1s = 5*np.arange(6,400)
###
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

        
        # Train using function+ Calculate residual for full training set:
        optimize_init = Optimizer(start_to_t1, eq_pop, no_states, initial_kappas=None)
        rms, optimized_kappas= optimize_init.run()
        save_kappas.append(optimized_kappas)
        save_resid1.append(rms)
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

np.savetxt(f"scan_0_5000_exc_resid1.dat", save_resid1, delimiter = "\t")
np.savetxt(f"scan_0_5000_exc_resid2.dat", save_resid2, delimiter = "\t")
np.savetxt(f"scan_0_5000_exc_resid3.dat", save_resid3, delimiter = "\t")
np.savetxt(f"scan_0_5000_exc_neghalf.dat", save_neg, delimiter = "\t")
np.savetxt(f"scan_0_5000_exc_kappas.dat", save_kappas, delimiter = "\t")
