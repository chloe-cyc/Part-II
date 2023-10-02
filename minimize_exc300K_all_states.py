#from scipy.optimize import minimize as min
from scipy.optimize import least_squares
from scipy.linalg import expm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DATA IMPORT - Using Pandas because it is easier to me
column_names = ["Time"]
excitation_data_path = "/u/dem/kebl6911/Part-II/mash_exc300K.dat"
for i in range(1,8):
    column_names.append(str(i))
# Over Time in ps
df = pd.read_csv(excitation_data_path, delimiter=" ", names=column_names)

#Selected columns to NP array of size 10001 x7
time_t = df["Time"].values

data_column_names = column_names[1:]
population_data = df[data_column_names]
population_data = population_data.values

#FIXED EQUILIBRIUM AND STARTING POPULATION
#Defining the equilibrium population as the final population, and extracting the initial population 
eq_pop = population_data[-1]
p_init = population_data[0]

#Optimizing w/ initial guess for k
initial_guess_k = np.array([0.001,0.001,0.001,0.001,0.001,0.001,0.001])
no_states = 7 #number of states

#Interchange between a flattened k and a k matrix
def list_to_matk(k): 
    
    matk = np.zeros((no_states,no_states))

    for i in range(no_states):
        for j in range(i+1,no_states):
             matk[i][j] = k[i]
             matk[j][i] = matk[i][j] 
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

#TEST OF K
p_test_result = []
for t in time_t:
    p_model_test = p_model(initial_guess_k,t)
    p_test_result.append(p_model_test)
p_test_result = np.array(p_test_result)
print(p_test_result)

#Plotting The results of the initial guess
count_plot =0

for column_name in column_names[1:]:
    plt.plot(time_t, df[column_name],".", label=column_name)
    plt.plot(time_t, p_test_result[:,count_plot],"-")
    count_plot+=1

# plt.plot(time_t, p_1_test_result,".")
# plt.plot(time_t, pop_1,".")

plt.legend()
plt.xlabel("Time(ps)")
plt.ylabel("Population (1-n)")
plt.show(block=True)

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

# def residuals(k):

#     p_model_result = np.array([p_model(k, ti) for ti in time_t])

#     return population_data[:,no_states]-p_model_result[:,no_states] #define the state for which we want 

# least_squares_result = least_squares(residuals, initial_guess_k, bounds=(np.array([1e-3]),np.array([1e3])))
# least_squares_result_min = least_squares_result.x

# p_test_result = []
# p_opt_result = []
# for t in time_t:
#     p_model_test = p_model(initial_guess_k, t)
#     p_model_opt = p_model(least_squares_result_min,t)
#     p_test_result.append(p_model_test)
#     p_opt_result.append(p_model_opt)
# p_test_result = np.array(p_test_result)
# p_opt_result = np.array(p_opt_result)
# # least_squares_data = np.column_stack(least_squares_data)

# file_names = ["model_result.dat", "least_squares_result.dat"]

# np.savetxt(file_names[1], p_test_result, delimiter = "\t",newline='\n')
# np.savetxt(file_names[0], p_opt_result, delimiter = "\t",newline='\n')


# #Plotting The results of the initial guess
# plt.plot(time_t, p_1_test_result,".")
# plt.plot(time_t, p_1_opt_result,"-")
# plt.plot(time_t, pop_1,".")
# legend = ["1","1_opt","actual1"]
# plt.legend(legend)
# plt.xlabel("Time(ps)")
# plt.ylabel("Population (1-n)")

# plt.show(block=True)