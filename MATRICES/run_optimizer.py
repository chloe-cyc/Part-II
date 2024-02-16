import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from optimagic_optimizer import Optimizer


data_path = "/u/dem/kebl6911/Part-II/MATRICES/Ct_site.out"

data = np.loadtxt(data_path)

t = data[:,0]
Ct = data[:,1:].reshape((-1,7,7))
Ct = np.einsum("tij->tji",Ct)

Cts = Ct[(np.arange(len(Ct)) % 10 == 0) | (np.arange(len(Ct)) == len(Ct) - 1)]
ts = t[(np.arange(len(t)) % 10 == 0) | (np.arange(len(t)) == len(t) - 1)]

t2 = 5000

#Extract equilibrium values of the populations of all sites from Cts
eq_pop = Ct[-1]
eq_pop = np.mean(eq_pop, axis=1)
no_states = 7
#only consider values below t2
t2_index = np.where(ts <= t2)[0] #currently testing
ts = ts[:t2_index[-1]+1] # Include t2
Cts = Cts[:t2_index[-1]+1]

save_kappas = []
save_resid1 = [] # Full Train
save_resid2 = [] # Half Train
save_resid3 = [] # Extrapolation
save_t = []
save_p1s = []
save_neg = []

t1s = 10*np.arange(3,200)
# t1s = np.array([int(200)])

for t1 in t1s:
    try:
        print(t1)
        save_t.append([t1])
        t0 = t1/2
        t0_index = np.where(ts >= t0)[0] #NEED TO USE 0 INDEX, NO NEED TO +1,
        t1_index = np.where(ts <= t1)[0] #NEED TO USE -1 INDEX +1 TO INDLUDE T1,

        start_to_t1_time = ts[:t1_index[-1]+1]
        start_to_t1_pop = Cts[:t1_index[-1]+1]

        #currently testing
        t0_to_t1_time = ts[t0_index[0]:t1_index[-1]+1] # Include t1
        t0_to_t1_pop = Cts[t0_index[0]:t1_index[-1]+1]

        t1_to_t2_time = ts[t1_index[-1]:]
        t1_to_t2_pop = Cts[t1_index[-1]:]

        # Train using function + Calculate residual for full training set:
        optimize_init = Optimizer(start_to_t1_time,start_to_t1_pop, eq_pop, no_states, initial_kappas=None)
        x, optimized_kappas= optimize_init.run()

        save_kappas.append(optimized_kappas)
        
        p0 = optimize_init.p0
        #Calculate Resid for Train set:
        predict_pop = optimize_init.predict(p0,start_to_t1_time,"back",optimized_kappas) #OUTPUT SHAPE??
        time_size = len(start_to_t1_time)
        
        resid1_single = start_to_t1_pop[:,:,0]-predict_pop[:,:,0]
        resid1_single = np.sqrt(np.sum(np.abs(resid1_single)**2)/(time_size*no_states))
        save_resid1.append(resid1_single)

        save_neg.append(optimize_init.time_size*7)

        # Calculate Residual for extrapolation set
        predict_pop = optimize_init.predict(p0,t1_to_t2_time,"foward",optimized_kappas) #OUTPUT SHAPE??
        time_size = len(t1_to_t2_time)
        
        print("hello")
        print(time_size)

        #ONLY FOR STARTING IN FIRST STATE!
    
        resid2_single = t1_to_t2_pop[:,:,0]-predict_pop[:,:,0]
        resid2_single = np.sqrt(np.sum(np.abs(resid2_single)**2)/(time_size*no_states))
        save_resid2.append(resid2_single)

        #Calculate Residual for Half training set
        predict_pop = optimize_init.predict(p0,t0_to_t1_time,"back",optimized_kappas)
        time_size = len(t0_to_t1_time)
        print(time_size)

        resid_half = t0_to_t1_pop[:,:,0] - predict_pop[:,:,0]
        resid3_single = np.sqrt(np.sum(np.abs(resid_half)**2)/(time_size*no_states))
        save_resid3.append(resid3_single)

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

np.savetxt(f"site_resid1.dat", save_resid1, delimiter = "\t") #0 to t1
np.savetxt(f"site_resid2.dat", save_resid2, delimiter = "\t") # t1 to end
np.savetxt(f"site_resid3.dat", save_resid3, delimiter = "\t") #0 to t1/2
np.savetxt(f"site_neghalf.dat", save_neg, delimiter = "\t")
np.savetxt(f"site_kappas.dat", save_kappas, delimiter = "\t")
