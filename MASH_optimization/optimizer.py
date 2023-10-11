import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def optimize(data, initial_kappas, eq_pop, full_time, full_population): #full_time, full_population

    def listkap_matkappa(kappa):
        matkappa = np.zeros((no_states,no_states))
        triangle = np.triu(np.ones((no_states, no_states), dtype=bool), k=1) 
        matkappa[triangle]= kappa
        matkappa += matkappa.T
        return matkappa

    def matkappa_matr(kappa):
        matkappa = listkap_matkappa(kappa)
        matr = np.zeros((no_states,no_states)) 
        matr = matkappa * eq_pop[:, np.newaxis] 
        rdiag = -np.sum(matr,0)
        matr = matr + np.diag(rdiag)
        return matr

    def p_model(exp_r_del_t):
        p_model = np.zeros((time_size,no_states))
        p_model[0,:] = init_pop
        for i in range(1,time_size):
            p_model[i,:] = exp_r_del_t.dot(p_model[i-1,:])
        return p_model

    def residuals(kappa):
        r_of_t = matkappa_matr(kappa)
        exp_r_del_t = expm(r_of_t*delta_t)
        p_model_result = p_model(exp_r_del_t) 
        residual = (population_data -p_model_result).flatten()
        # residual = population_data -p_model_result
        return residual

    #Inputs 
    time = data[:,0]
    population_data = data[:,1:]
    delta_t = time[1]-time[0]
    no_states = len(population_data[0])
    init_pop = population_data[0]
    time_size = time.size

    #Optimization
    #Enforce kappa>0
    least_squares_result = least_squares(residuals, initial_kappas, bounds=(0,float("inf"))) # bounds=(-float("inf"),1e-17)
    optimized_kappa = least_squares_result.x
    print(optimized_kappa)
    # for kappa in optimized_kappa_1:
    #     least_squares_bounded_result = least_squares(residuals, kappa, bounds = (-1.))
        
    optimized_r = matkappa_matr(optimized_kappa)
    optimized_eigenvalues = np.linalg.eig(optimized_r)[0]
    # print(optimized_eigenvalues)

    #Calculation of statepopulation for output
    population_data = full_population[:,1:]
    time_size = full_time.size
    r_of_t_ls = matkappa_matr(optimized_kappa)
    exp_ls_del_t = expm(r_of_t_ls*delta_t)
    optimized_population = np.array(p_model(exp_ls_del_t))
    time = full_time[:,np.newaxis]
    # loss = sum(residuals(optimized_kappa)**2)
    # print(least_squares_result.status)
    optimized_population = np.concatenate((time, optimized_population), axis=1)


    return optimized_population, optimized_eigenvalues, optimized_kappa

