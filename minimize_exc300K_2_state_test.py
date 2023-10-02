#from scipy.optimize import minimize as min
from scipy.optimize import least_squares
from scipy.linalg import expm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Optimizing w/ initial guess for k
initial_guess_k = np.array([-0.01])
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
    matr = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matr[i][j] = matk[i][j]*eq_pop[i] # I want i !=j to be completed first 

    for h in range (n): #let matrix ij be also written as jh to avoid clashing in the same defined equation
        matr[h][h] = -sum(matr[j][h]*eq_pop[j] for j in range(n) if j != h) 
    return matr

def p_model(k, t):
    r_of_t = matk_to_matr(k)
    #print(np.linalg.eig(r_of_t))
    r_t = r_of_t*t
    exp_rt = expm(r_t)
    p_t = exp_rt.dot(p_init)
    return p_t

#Population Vectors
time_t = np.arange(0, 1001,1) 
eq_pop=np.array([0.6, 0.4])
k21 =np.array([-0.2])
real_rate = ([-k21*eq_pop[1],k21*eq_pop[0]],[k21*eq_pop[1],-k21*eq_pop[0]])
p_init = np.array([1,0])

population_data = []

for t in time_t:
    pop_test = p_model(k21,t)
    population_data.append(pop_test)

population_data = np.array([population_data])
print(population_data)


def residuals(k):
    p_model_result = np.array([p_model(k, ti) for ti in time_t])
    return ((population_data -p_model_result)**2).flatten()

least_squares_result = least_squares(residuals, initial_guess_k, method="lm")
least_squares_result_min = least_squares_result.x
print(least_squares_result_min)

# # minimized_result = min(function_to_minimize,initial_guess_k)
# # minimized_k = minimized_result.x1)
# p_least_squares = []
# p_test_result = []
# for t in time_t:
#     p_model_test = p_model(initial_guess_k,t)
#     p_test_result.append(p_model_test)
#     p_model_least_squares = p_model(least_squares_result,t)
#     p_least_squares.append(p_model_least_squares)
# p_test_result = np.array(p_test_result)
# p_least_squares = np.array(p_least_squares)

# column_names = ["Time", "1", "2"]

# for i,column_name in enumerate(column_names[2:],start=1):
#     c = 'C%i'%i
#     plt.plot(time_t, population_data,":", label=column_name, color=c)
#     plt.plot(time_t, p_test_result[:, i],"-", color=c,label=f"_test")
#     plt.plot(time_t, p_least_squares[:,i],"--",color=c, label=f"_ls")

# plt.legend()
# plt.xlabel("Time(ps)")
# plt.ylabel("Population (1-n)")
# plt.show(block=True)