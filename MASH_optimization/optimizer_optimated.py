import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def runmodel(kappas, p_init, time):
    
    """
    kappas: kappa values for the current model, 1D array
    p_init: Initial population at time[0], 1D array of size N
    time: times at which to calculate populations
    """
    n = len(kappa_size)
    kappa_size = (n*(n-1))/2
    if kappa_size != len(kappas):
        return "Incompatible sizing"
    
    else:

        return 


class optimizer:
    def __init__(self, data, initial_kappas=None, eq_pop, full_time): #full_time, full_population
        self.data = data
        self.initial_kappas = initial_kappas
        self.eq_pop = eq_pop
        self.full_time = full_time

    def listkap_matkappa(self, kappa):
        matkappa = np.zeros((no_states,no_states))
        triangle = np.triu(np.ones((no_states, no_states), dtype=bool), k=1) 
        matkappa[triangle]= kappa
        matkappa += matkappa.T
        return matkappa

    def matkappa_matr(self, kappa):
        matkappa = listkap_matkappa(kappa)
        matr = np.zeros((no_states,no_states)) 
        matr = matkappa * eq_pop[:, np.newaxis] 
        rdiag = -np.sum(matr,0)
        matr = matr + np.diag(rdiag)
        return matr

    def p_model(self, exp_r_del_t):
        p_model = np.zeros((time_size,no_states))
        p_model[0,:] = init_pop
        for i in range(1,time_size):
            p_model[i,:] = exp_r_del_t.dot(p_model[i-1,:])
        return p_model

    def residuals(self, kappa):
        r_of_t = matkappa_matr(kappa)
        exp_r_del_t = expm(r_of_t*delta_t)
        p_model_result = p_model(exp_r_del_t)
        residual = (population_data -p_model_result).flatten()
        return residual
    
    def optimize(self):
        # Extraction of data
        self.time = self.data[:,0]
        self. population_data = self.data[:,1]
        self.delta_t = 
        return residual_sum_divided, optimized_population

    
    time = data[:,0]
    population_data = data[:,1:]
    delta_t = time[1]-time[0]
    no_states = len(population_data[0])
    init_pop = population_data[0]
    time_size = time.size
    data_size = time_size
    
    #Optimization
    least_squares_result = least_squares(residuals, initial_kappas, bounds=(0,float("inf"))) # bounds=(-float("inf"),1e-17)
    optimized_kappa = least_squares_result.x

    # Calculation of state population for output
    #time_size = full_time.size
    r_of_t_ls = matkappa_matr(optimized_kappa)
    exp_ls_del_t = expm(r_of_t_ls*delta_t)
    optimized_population = np.array(p_model(exp_ls_del_t))
    #full_time = full_time[:,np.newaxis]
    full_time = time[:,np.newaxis]
    
    # loss = sum(residuals(optimized_kappa)**2)
    # print(least_squares_result.status)
    residual_sum_divided = sum(abs(residuals(optimized_kappa)))/data_size
    optimized_population = np.concatenate((full_time, optimized_population), axis=1)


    
