#from scipy.optimize import minimize as min
from scipy.optimize import least_squares
from scipy.linalg import expm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# DATA IMPORT
excitation_data_path = "/u/dem/kebl6911/Part-II/mash_exc300K.dat"
column_names = ["Time", "1"]  # Over Time in ps
df = pd.read_csv(excitation_data_path, delimiter=" ",usecols=[0, 1], names=column_names)
# Conversion to np arrays
time_t = df["Time"].values
pop_1 = df["1"].values

#FIXED EQUILIBRIUM AND STARTING POPULATION
#Defining the equilibrium population as the final population, 
equilibrium_pop_1 = pop_1[-1]
#Defining the initial population (1) as the total population
total_pop = pop_1[0]
#Defining the population of 2. over time
equilibrium_pop_2=total_pop - equilibrium_pop_1
pop_2 = total_pop-pop_1

#Population Vectors
eq_pop=np.array([equilibrium_pop_1, equilibrium_pop_2])
#stack this population_vec = [pop_1, pop_2]
population_data = []
for i in range (len(pop_1)):
    column = np.array([[pop_1[i]],[pop_2[i]]])
    population_data.append(column)

population_data = np.array(population_data) #Shape 10001 x n x 1
population_data = population_data[:,:,0]
#Initial population P(0)
p_init = population_data[1]

#Optimizing w/ initial guess for k
initial_guess_k = np.array([0.001])
n = 2 #number of states


#Interchange between a flattened k and a k matrix
def list_to_matk(k): 
    
    matk = np.zeros((n,n))

    index = 0
    for i in range(n):
        for j in range(i+1,n):
             matk[i][j] = k[index]
             matk[j][i] = matk[i][j] 
             index +=1
    return matk

def matk_to_matr(k):
    matk = list_to_matk(k)
    n = int(matk.shape[0])
    matr = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matr[i][j] = matk[i][j]*eq_pop[i] # I want i !=j to be completed first 

    for h in range (n): #let matrix ij be also written as jh to avoid clashing in the same defined equation
        matr[h][h] = -sum(matr[j][h] for j in range(n) if j != h) 
    return matr

def p_model(k, t):
    r_of_t = matk_to_matr(k)
    #print(np.linalg.eig(r_of_t))
    r_t = r_of_t*t
    exp_rt = expm(r_t)
    p_t = exp_rt.dot(p_init)
    return p_t


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
# plt.show

states = 1
def residuals(k):

    p_model_result = np.array([p_model(k, ti) for ti in time_t])
    return population_data[:,states,0]-p_model_result[:,states,0]



least_squares_result = least_squares(residuals, initial_guess_k, bounds=(np.array([1e-3]),np.array([1e3])))
least_squares_result_min = least_squares_result.x
# minimized_result = min(function_to_minimize,initial_guess_k)
# minimized_k = minimized_result.x

#Testing that the functions work
p_test_result = []
p_opt_result = []
for t in time_t:
    p_model_test = p_model(initial_guess_k, t)
    p_model_opt = p_model(least_squares_result_min,t)
    p_test_result.append(p_model_test)
    p_opt_result.append(p_model_opt)
p_test_result = np.array(p_test_result)
p_opt_result = np.array(p_opt_result)

print(p_test_result)

p_1_test_result = p_test_result[:, 0]
p_1_opt_result = p_opt_result[:,0]

#Plotting The results of the initial guess
plt.plot(time_t, p_1_test_result,".")
plt.plot(time_t, p_1_opt_result,".")
plt.plot(time_t, pop_1,".")
legend = ["1","1_opt","actual1"]
plt.legend(legend)
plt.xlabel("Time(ps)")
plt.ylabel("Population (1-n)")

plt.show(block=True)