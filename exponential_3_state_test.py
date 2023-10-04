#from scipy.optimize import minimize as min
from scipy.optimize import least_squares
from scipy.linalg import expm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def listk_to_matr(k):
    index=0
    mat_k = np.zeros((no_states,no_states))
    for i in range(no_states):
        for j in range(no_states):
            if i != j:
                mat_k[i][j] = k[index]
                index +=1
    for h in range(no_states):
        mat_k[h][h] = -sum(mat_k[f][h] for f in range(no_states) if f != h)
    return mat_k

#Functions for optimization in terms of Kappa
def listkappa_to_matkappa(kappa):  
    matkappa = np.zeros((no_states,no_states))
    index = 0
    for i in range(no_states):
        for j in range(i+1,no_states):
            matkappa[i][j] = kappa[index]
            matkappa[j][i] = matkappa[i][j] 
            index +=1
    return matkappa

def matkappa_to_matr(kappa):
    matk = listkappa_to_matkappa(kappa)
    matr = np.zeros((no_states,no_states))
    for i in range(no_states):
        for j in range(no_states):
            if (i != j) ==True:
                matr[i][j] = matk[i][j]*eq_pop[i] # I want i !=j to be completed first 

    for h in range (no_states): #let matrix ij be also written as fh to avoid clashing in the same defined equation
        matr[h][h] = -sum(matr[f][h] for f in range(no_states) if f != h) 
    return matr

def p_model(kappa, t):
    r_of_t = matkappa_to_matr(kappa)
    #print(np.linalg.eig(r_of_t))
    r_t = r_of_t*t
    exp_rt = expm(r_t)
    p_t = exp_rt.dot(p_init)
    return p_t

#Calculate test-population using k values
k_array = np.array([0.9,0.13,0.14,0.15,0.16,2])
p_init= np.array([1,0,0])
no_states = 3 #number of states
mat_r_k = listk_to_matr(k_array)
print(np.linalg.eig(mat_r_k))  

#Generation of exponential data from k using P = e^Rt
time_t = np.arange(0, 50,0.1)
population_data = []
for t in time_t:
    exp_rt = expm(mat_r_k*t).dot(p_init)
    population_data.append(exp_rt)
population_data = np.array(population_data)

#equilibrium population
eq_pop = population_data[-1]
print(eq_pop)

def residuals(kappa):
    p_model_result = np.array([p_model(kappa, ti) for ti in time_t])
    return ((population_data-p_model_result)**2).flatten()

#Optimization
kappa_to_test = np.array([0.01,0.01,0.01])
least_squares_result = least_squares(residuals, kappa_to_test, method="lm")
least_squares_kappa_result = np.array(least_squares_result.x)
print(f"this is the test k {mat_r_k}")
print(f"this is ls result in terms of kappa {least_squares_kappa_result}")
least_squares_kappa_rmatrix = matkappa_to_matr(least_squares_kappa_result)

print(f"this is the k matrix computed from kappa {least_squares_kappa_rmatrix }")


p_least_squares = []
# p_test_result = []
for t in time_t:
#     p_model_test = p_model(kappa_to_test,t)
#     p_test_result.append(p_model_test)
    p_model_least_squares = p_model(least_squares_kappa_result,t)
    p_least_squares.append(p_model_least_squares)
# p_test_result = np.array(p_test_result) 
p_least_squares = np.array(p_least_squares)
# column_names = ["Time", "1", "2"]



#Plotting
c=1
for i in range(no_states):
    plt.plot(time_t, population_data[:,i],":")
    # plt.plot(time_t, p_test_result[:, i],"-", color=colors,label=f"_test")
    plt.plot(time_t, p_least_squares[:,i],"--", label=f"_ls")


plt.legend()
plt.xlabel("Time(ps)")
plt.ylabel("Population (1-n)")
plt.show(block=True)

# # for i in range(i):
# #     c="C%i%i"
# #     plt.plot(time_t, population_data[:,i],":", label=column_names, color=c)
# #     plt.plot(time_t, p_test_result[:, i],"-", color=c,label=f"_test")
# #     plt.plot(time_t, p_least_squares[:,i],"--",color=c, label=f"_ls")
