#from scipy.optimize import minimize as min
from numba import jit,prange
from scipy.optimize import least_squares
from scipy.linalg import expm
import numpy as np
import pandas as pd
import time


timer_start = time.time()

def new_listkappa_to_matkappa(kappa):
    matkappa = np.zeros((no_states,no_states))
    triangle = np.triu(np.ones((no_states, no_states), dtype=bool), k=1) #np.triu is a command that specifies the upper triangle of the matrix without the diagonal(k=1)
    matkappa[triangle]= kappa
    matkappa += matkappa.T
    return matkappa

def new_matkappa_to_matr(kappa):
    matkappa = new_listkappa_to_matkappa(kappa)
    matr = np.zeros((no_states,no_states)) 
    #for j in range(no_states):
     #   matr[:,j] = matkappa[:,j]*eq_pop[:]
    matr = matkappa * eq_pop[:, np.newaxis] 
    #elementwisemultiplication between two np arrays, reshaping eq_pop from 1D to 2D col vector allows for broadcasting during elementwise multiplication
    rdiag = -np.sum(matr,0)
    matr = matr + np.diag(rdiag)
    return matr

def p_model(exp_r_del_t,n):
    # exp_rt = np.linalg.matrix_power(exp_r_del_t,n)
    p_model = np.zeros((n,no_states))
    p_model[0,:] = p_init
    # exp_curr = exp_r_del_t
    for i in range(1,n):
        p_model[i,:] = exp_r_del_t.dot(p_model[i-1,:])
    # p_t = exp_rt.dot(p_init)
    return p_model

def residuals(kappa):
    r_of_t = new_matkappa_to_matr(kappa)
    exp_r_del_t = expm(r_of_t*delta_t)
    p_model_result = p_model(exp_r_del_t,time_t.size) #np.array([p_model(exp_r_del_t,n) for n in range(time_t)])
    residual = (abs(population_data -p_model_result)**2).flatten()
    return residual    #return population_data[:,1]-p_model_result[:,1] #define the state for which we want 

# DATA IMPORT - Using Pandas because it is easier to me
column_names = ["Time"]
site_data_path = "Data/mash_site300K.dat"
for i in range(1,8):
    column_names.append(str(i))

df = pd.read_csv(site_data_path, delimiter=" ", names=column_names)
#df = df[df["Time"]<=2000]
#df = df[df["Time"]>500]
#df = df[(df.index % 20 == 0) | (df.index == len(df) - 1)]
data_column_names = column_names[1:]
delta_t = df.iloc[1, 0] - df.iloc[0, 0]

#Start time
start_time_values = np.array(df['Time'].iloc[1::2])
start_time_values = start_time_values[start_time_values >=1000]

#Optimizing w/ initial guess for k
no_states = 7 #number of states
num_elements = no_states*(no_states-1)//2
initial_guess_kappa = np.full(num_elements,0.01)

residual_sum = []
kappa_saved =[]

for index, start_time in enumerate(start_time_values, start=1):
    #Only consider values that are larger than the start_time
    data_column_names = column_names[1:]
    df = df[df["Time"]>=start_time]
    population_data = df[data_column_names].values
    time_t = df["Time"].values
    

    #FIXED EQUILIBRIUM AND STARTING POPULATION
    #Defining the equilibrium population as the final population, and extracting the initial population 
    eq_pop = population_data[-1]
    p_init = population_data[0]

    least_squares_result = least_squares(residuals, initial_guess_kappa, jac = "2-point", method="lm")
    least_squares_result = least_squares_result.x
    
    residual_sum.append(sum(residuals(least_squares_result)))
    kappa_saved.append(least_squares_result)
    #kappa_saved.append(index, least_squares_result)
    print(f"run {index}")

print(kappa_saved)
residual_data = np.column_stack((start_time_values,residual_sum))
np.savetxt(f"Site_data/residual_data.dat", residual_data,delimiter="\t")
np.savetxt(f"Site_data/least_squares_kappas.dat", kappa_saved, delimiter="\t")
timer_end = time.time()
excecution_time =  timer_end-timer_start
print(f"excecution time {excecution_time}")

