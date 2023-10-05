#from scipy.optimize import minimize as min
from scipy.optimize import least_squares
from scipy.linalg import expm
import numpy as np
import pandas as pd
import time


start_time = time.time()

# DATA IMPORT - Using Pandas because it is easier to me
column_names = ["Time"]
excitation_data_path = "/u/dem/kebl6911/Part-II/MASH_optimization/Data/mash_exc300K.dat"
for i in range(1,8):
    column_names.append(str(i))
    
# Over Time in ps
df = pd.read_csv(excitation_data_path, delimiter=" ", names=column_names)
#df = df[df["Time"]<=000]
#df = df[df["Time"]>500]
df = df[(df.index % 50 == 0) | (df.index == len(df) - 1)]

#Selected columns to NP array of size 10001 x7
time_t = df["Time"].values
delta_t = time_t[1]-time_t[0]

data_column_names = column_names[1:]
population_data = df[data_column_names]
population_data = population_data.values

#FIXED EQUILIBRIUM AND STARTING POPULATION
#Defining the equilibrium population as the final population, and extracting the initial population 
eq_pop = population_data[-1]
# eq_pop = np.array([1,2,3])
p_init = population_data[0]

#Optimizing w/ initial guess for k
no_states = 7 #number of states
num_elements = no_states*(no_states-1)//2
initial_guess_kappa = np.full(num_elements,0.01)
#initial_guess_kappa = np.array([1,2,3])

#Interchange between a flattened k and a k matrix

def new_listkappa_to_matkappa(kappa):
    matkappa = np.zeros((no_states,no_states))
    triangle = np.triu(np.ones((no_states, no_states), dtype=bool), k=1) #np.triu is a command that specifies the upper triangle of the matrix without the diagonal(k=1)
    matkappa[triangle]= kappa
    matkappa += matkappa.T
    return matkappa

# print(new_listkappa_to_matkappa(initial_guess_kappa))
# print(timeit.timeit(lambda: new_listkappa_to_matkappa(initial_guess_kappa), number=1))

# print(listkappa_to_matkappa(initial_guess_kappa))
# print(timeit.timeit(lambda: listkappa_to_matkappa(initial_guess_kappa), number = 1))

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

# def old_matkappa_to_matr(kappa):
#     matkappa = listkappa_to_matkappa(kappa)
#     matr = np.zeros((no_states,no_states))
#     for i in range(no_states):
#         for j in range(no_states):
#             matr[i][j] = matkappa[i][j]*eq_pop[i] # I want i !=j to be completed first
#     for h in range(no_states): #let matrix ij be also written as jh to avoid clashing in the same defined equation
#         matr[h][h] = -sum(matr[j][h] for j in range(no_states) if j != h) # splice?
#     return matr


def p_model(exp_r_del_t,n):
    # exp_rt = np.linalg.matrix_power(exp_r_del_t,n)
    p_model = np.zeros((n,no_states))
    p_model[0,:] = p_init
    # exp_curr = exp_r_del_t
    for i in range(1,n):
        p_model[i,:] = exp_r_del_t.dot(p_model[i-1,:])
    # p_t = exp_rt.dot(p_init)
    return p_model


# def function_to_minimize(k):
#     sum=0
#     for i in range(len(population_data[0])): # need to take range from the number of rows there are int he column vector.
#         p_model_result = p_model(k, time_t[i])
#         for state in range(n):
#             run = abs(population_data[state][i] - p_model_result[state])**2
#             sum += run
#     return sum
# ks = np.logspace(-3,3)
# fs = [function_to_minimize(np.array([k])) for k in ks]
# plt.semilogx(ks,fs,".")

def residuals(kappa):
    r_of_t = new_matkappa_to_matr(kappa)
    exp_r_del_t = expm(r_of_t*delta_t)
    p_model_result = p_model(exp_r_del_t,time_t.size) #np.array([p_model(exp_r_del_t,n) for n in range(time_t)])
    residual = ((population_data -p_model_result)**2).flatten()
    return residual
    #return population_data[:,1]-p_model_result[:,1] #define the state for which we want 

least_squares_result = least_squares(residuals, initial_guess_kappa, jac = "2-point", method="lm")
least_squares_result = least_squares_result.x
print(least_squares_result)


#TEST OF K
# p_test_result = []
# p_least_squares = []

#p_model_test = p_model(initial_guess_kappa,t)
#p_test_result.append(p_model_test)
r_of_t_ls = new_matkappa_to_matr(least_squares_result)
exp_ls_del_t = expm(r_of_t_ls*delta_t)
p_model_least_squares = np.array(p_model(exp_ls_del_t,time_t.size))

# # # #p_test_result = np.array(p_test_result)
# # p_least_squares = np.array(p_least_squares)
# np.savetxt("exc_ls.dat",p_model_least_squares, delimiter="\t")

end_time = time.time()
excecution_time =  end_time-start_time
print(f"excecution time {excecution_time}")
# ------------------------------------------------------------------------------------------------

# #Plotting The results of the initial guess

# print(p_least_squares)
# for i,column_name in enumerate(column_names[2:],start=1):
#     c = 'C%i'%i
#     plt.plot(time_t, df[column_name],":", label=column_name, color=c)
#     #plt.plot(time_t, p_test_result[:, i],"-", color=c,label=f"_test")
#     plt.plot(time_t, p_least_squares[:,i],"--",color=c, label=f"{column_name}_ls")

# plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
# plt.xlabel("Time(ps)")
# #plt.xlim(500,4000)
# plt.ylabel("Population (1-n)")
# plt.show(block=True)

# p_test_result = []
# p_opt_result = []
# for t in time_t:
#     p_model_test = p_model(initial_guess_kappa, t)
#     p_model_opt = p_model(least_squares_result_min,t)
#     p_test_result.append(p_model_test)
#     p_opt_result.append(p_model_opt)
# p_test_result = np.array(p_test_result)
# p_opt_result = np.array(p_opt_result)
# # least_squares_data = np.column_stack(least_squares_data)

# file_names = ["model_result.dat", "least_squares_result.dat"]

# np.savetxt(file_names[1], p_test_result, delimiter = "\t",newline='\n')
# np.savetxt(file_names[0], p_opt_result, delimiter = "\t",newline='\n')


# #Plotting The results of the initial guessd
# plt.plot(time_t, p_1_test_result,".")
# plt.plot(time_t, p_1_opt_result,"-")
# plt.plot(time_t, pop_1,".")
# legend = ["1","1_opt","actual1"]
# plt.legend(legend)
# plt.xlabel("Time(ps)")
# plt.ylabel("Population (1-n)")

# plt.show(block=True)