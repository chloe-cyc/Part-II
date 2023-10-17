#from scipy.optimize import minimize as min
from scipy.optimize import least_squares
from scipy.linalg import expm
import numpy as np
import pandas as pd
import time
start_time = time.time()

# DATA IMPORT - Using Pandas because it is easier to me
column_names = ["Time"]
site_data_path = "/u/dem/kebl6911/Part-II/MASH_optimization/Data/mash_site300K.dat"
for i in range(1,8):
    column_names.append(str(i))
    
# Over Time in ps
site_df = pd.read_csv(site_data_path, delimiter=" ", names=column_names)
#df = df[df["Time"]<=]
site_time_cutoff_df =site_df[site_df["Time"]>=4.5e+02] #new df to retain old one
site_time_cutoff_df = site_time_cutoff_df[(site_time_cutoff_df.index % 30 == 0) | (site_time_cutoff_df.index == len(site_time_cutoff_df.index) - 1)]

# site_data = site_df.values
# site_data_time = site_data[:,0]
# population_data = site_data[:,1:]
# print(population_data)


#Selected columns to NP array of size 10001 x7
time_t = site_time_cutoff_df["Time"].values
delta_t = time_t[1]-time_t[0]

data_column_names = column_names[1:]
population_data = site_time_cutoff_df[data_column_names]
population_data = population_data.values


# #FIXED EQUILIBRIUM AND STARTING POPULATION
# #Defining the equilibrium population as the final population, and extracting the initial population 
eq_pop = population_data[-1]
p_init = population_data[0]
print(len(population_data[0]))

#Optimizing w/ initial guess for k
no_states = 7#number of states
num_elements = no_states*(no_states-1)//2
initial_guess_kappa = np.full(num_elements,0.01)

#Interchange between a flattened k and a k matrix

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
    residual = ((population_data -p_model_result)**2).flatten()
    return residual
    #return population_data[:,1]-p_model_result[:,1] #define the state for which we want 

#least_squares_result = least_squares(residuals, initial_guess_kappa, jac = "3-point", method="lm")
least_squares_result = least_squares(residuals, initial_guess_kappa, method="lm")
least_squares_result = least_squares_result.x
print(least_squares_result)

#TEST OF K
# p_test_result = []
# p_least_squares = []

#p_model_test = p_model(initial_guess_kappa,t)
#p_test_result.append(p_model_test)
# # #p_test_result = np.array(p_test_result)
# p_least_squares = np.array(p_least_squares)

#BACKPROPOGATION CALCULATION
# site_df = site_df[(site_df.index % 30 == 0) | (site_df.index == len(site_df.index) - 1)]

# data_column_names = column_names[1:]
# population_data = site_df[data_column_names]
# population_data = population_data.values

# #FIXED EQUILIBRIUM AND STARTING POPULATION
# #Defining the equilibrium population as the final population, and extracting the initial population 
# eq_pop = population_data[-1]
# p_init = population_data[0]


r_of_t_ls = new_matkappa_to_matr(least_squares_result)
exp_ls_del_t = expm(r_of_t_ls*delta_t)
p_model_least_squares = np.array(p_model(exp_ls_del_t,time_t.size))
np.savetxt("/u/dem/kebl6911/Part-II/MASH_optimization/site_ls.dat",p_model_least_squares, delimiter="\t")


end_time = time.time()
excecution_time =  end_time-start_time
print(f"excecution time {excecution_time}")
# # ------------------------------------------------------------------------------------------------
