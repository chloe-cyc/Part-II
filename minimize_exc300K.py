from scipy.optimize import minimize as min
from scipy.linalg import expm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

start_time = time.time()

def matk_to_list(matrix):
    singlerow_k = []
    n = matrix.shape[0]

    for i in range(n): # n is the shape of the matrix    
        for j in range(i+1,n):
            singlerow_k.append(matrix[i,j])
    return singlerow_k # for the matrix above this would return a value k12
#TEST 
n = 2
row_k = np.arange(n*(n-1)/2)+1

#Interchange between a flattened k and a k matrix
def list_to_matk(row_k, n): #
    mat = np.zeros((n,n))

    index = 0
    for i in range(n):
        for j in range(i+1,n):
             mat[i][j] = row_k[index]
             mat[j][i] = mat[i][j] 
             index +=1
    return mat
test_k =list_to_matk(row_k,n)


def matk_to_matr(matk, eq_pop):
    n = int(matk.shape[0])
    matr = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matr[i][j] = matk[i][j]*eq_pop[i] # I want i !=j to be completed first 

    for h in range (n): #let matrix ij be also written as jh to avoid clashing in the same defined equation
        matr[h][h] = -sum(matr[j][h] for j in range(n) if j != h) 
    return matr
eq_pop = np.array([1,1,1,1])
r = matk_to_matr(test_k,eq_pop)
#print(np.sum(r,1))

#Defining test population
#test_r = np.array([[1,1],[1,1]])
eigenvalues, eigenvectors = np.linalg.eig(r)
#print(eigenvalues, eigenvectors)
test_t = np.array([0,1,2])
p_1 = np.array([[1], [0]])  # 2x1 matrix
p_2 = np.array([[0.5], [0.5]])  # 2x1 matrix
p_3 = np.array([[0], [1]])  # 2x1 matrix
test_p = np.array([p_1, p_2, p_3] ) # List of 2x1 matrices
p_init = test_p[0]
#print(p_init)

#Defining Model P  - w/o for loop
def p_model(r_of_t, t, p_init):
    r_t = r_of_t*t
    exp_rt = expm(r_t)
    p_t = exp_rt.dot(p_init)
    return p_t

#Defining Model P  - w/o for loop
p_init = #global var
eq_pop = #global var
def p_model(k, t):
    r_of_t = matk_to_matr(k,eq_pop)
    r_t = r_of_t*t
    exp_rt = expm(r_t)
    p_t = exp_rt.dot(p_init)
    return p_t
    
# Using Model P with a Loop outside - example
p_test_result = []
for t in test_t:    
    p_test = p_model(r, t, p_init)
    p_test_result.append(p_test)
p_test_result = np.array(p_test_result)
#print(p_test_result) 


#def p_model(rmatrix, t_data, p_init):  # where k is calculated from the code above
#    r_t = np.array([rmatrix*t for t in t_data])
 #   print(f"this is {r_t}")
  #  exp_rt = np.array([expm(matrix) for matrix in r_t])
  #  print(f"this is {exp_rt}")
  #  p_t = [exp_matrix.dot(p) for exp_matrix, p in zip(exp_rt, p_data)]
   # print(f"this is {p_t}")
    #return p_t
#print(p_model(test_r,test_t,test_p))

#Defining the Function to Minimize 
def function_to_minimize(p_model, p_data):
    sum = 0
    vector_height = p_model[0].shape[0]
    for i in range(len(p_data[0])): # need to take range from the number of rows there are int he column vector.
        for n in range(vector_height):
            run = abs(p_data[n][i, 0] - p_model[n][i, 0])**2
            sum += run 
    return sum
#print(function_to_minimize(test_p, test_p1)) 


p_data = #TBD
def function_to_minimize(k):

    sum = 0
    vector_height = p_model[0].shape[0]
    for i in range(len(p_data[0])): # need to take range from the number of rows there are int he column vector.
        
        p_model_result = p_model(k,t[i])
        for n in range(vector_height):
            run = abs(p_data[n][i] - p_model_result[n])**2
            sum += run 
    return sum
#print(function_to_minimize(test_p, test_p1)) 

#print(function_to_minimize(test_p, test_p1)) 
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
eq_population_vec=np.array([equilibrium_pop_1, equilibrium_pop_2])
#stack this population_vec = [pop_1, pop_2]
population_data = []
for i in range (len(pop_1)):
    column = np.array([[pop_1[i]],[pop_2[i]]])
    population_data.append(column)
population_data = np.array(population_data)

#Optimizing w/ initial guess for k
initial_guess = np.array([0.01])

#Creating the R matrix
test_matk = list_to_matk(initial_guess,2)
#print(test_matk)
test_matr = matk_to_matr(test_matk,eq_population_vec)
eigenvalues, eigenvectors = np.linalg.eig(test_matr)
print(eigenvalues, eigenvectors)
#print(test_matr)
p_init_test = population_data[1]

#For loop to run p_model_test:
p_test_result = []
for t in time_t:
    p_model_test = p_model(test_matr, t, p_init_test)
    p_test_result.append(p_model_test)
p_test_result = np.array(p_test_result)
print(p_test_result)


#print(p_model_test)
print(time.time() - start_time)

p_1_test_result = p_test_result[:, 0, 0]
p_2_test_result = p_test_result[:, 1, 0]


#Plotting The results of the initial guess
plt.plot(time_t, p_1_test_result,".")
plt.plot(time_t, p_2_test_result,".")
plt.plot(time_t, pop_1,".")
legend = ["1","2","actual1"]
plt.legend(legend)
plt.xlabel("Time(ps)")
plt.ylabel("Population (1-n)")
#plt.legend()

plt.show(block=True)


#MINIMIZATION
#result = min(, initial_guess, method="BFGS")
#optimal_k = result.x
