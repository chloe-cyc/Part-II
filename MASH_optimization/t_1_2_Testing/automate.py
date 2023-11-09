import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import pandas as pd
import optimizer_optimated

column_names = ["Time"]
site_data_path = "Data/mash_site77K.dat"
for i in range(1,8):
    column_names.append(str(i))
site_df = pd.read_csv(site_data_path, delimiter=" ", names=column_names)
site_df = site_df[(site_df.index % 10 == 0) | (site_df.index == len(site_df.index) - 1)]
site_data = site_df.values

dt1 = 50
# t1s = dt1*np.arange(1,4000//dt1)
t1s = dt1*np.arange(1,4000//dt1)

eq_pop = site_df.values[-1,1:]
ns = len(eq_pop)
initial_kappas = np.zeros(ns*(ns-1)//2)

fnam = "rmse.dat"
f = open(fnam,'w')
f.close()

crit1 = False
crit2 = False
error_prev = 1.
kappa_prev = np.zeros_like(initial_kappas)
t1_prev = t1s[0]

for t1 in t1s:
    # Only values below t1
    ls_data = site_df[site_df["Time"]<=t1]
    ls_data = ls_data.values #DF to NP

    optimizer = optimizer_optimated.Optimizer(ls_data,eq_pop,ns,initial_kappas=initial_kappas)
    res,kappa_opt = optimizer.run()

    p0 = ls_data[-1,1:]
    time_left = ls_data[ls_data[:,0] > t1/2][:,0]
    exact_left = ls_data[ls_data[:,0] > t1/2][:,1:]
    pop_left = optimizer.predict(p0,time_left,direction='back',kappa=kappa_opt)
    error = np.sqrt(np.sum((exact_left-pop_left)**2)/(len(exact_left)*ns))

    print(t1,error)

    if (error < 0.01):
        crit1 = True
    else:
        crit1 = False

    if crit1 and (error > error_prev):
        crit2 = True

    if (crit1 and crit2):
        t1 = t1_prev
        kappa_opt = kappa_prev
        error = error_prev
        print("Optimum reached for t1 = ",t1,", plotting populations.")
        print("Final RMSE: ",error)
        ls_data = site_data[site_data[:,0]<=t1]
        p0 = ls_data[-1,1:]
        time_left = ls_data[:,0]
        time_right = site_data[site_data[:,0] > t1][:,0]
        pop_left = optimizer.predict(p0,time_left,direction='back',kappa=kappa_opt)
        pop_right = optimizer.predict(p0,time_right,direction='forward',kappa=kappa_opt)
        pred_time = np.concatenate((time_left,time_right))
        pred_pop = np.concatenate((pop_left,pop_right),axis=0)
        plt.figure()
        for i in range(7):
            c = "C%i"%i
            plt.plot(site_data[:,0],site_data[:,i+1],'-',color=c)
            plt.plot(pred_time,pred_pop[:,i],'--',color=c)
        plt.show()

        # write to disk
        save = np.column_stack((pred_time,pred_pop))
        np.savetxt("mash_site300K_opt.dat",save,delimiter="           ")
        break

    error_prev = error
    kappa_prev = kappa_opt
    t1_prev = t1

