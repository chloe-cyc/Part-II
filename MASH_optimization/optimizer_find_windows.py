import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def optimize(data, initial_kappas, initial_p_zero, eq_pop, full_time, t_one): #t_one, t_two, full_time): #full_time, full_population

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

    def p_model(exp_r_del_t, p_zero):
        p_model = np.zeros((time_size,no_states))
        p_model[0,:] = p_zero
        for i in range(1,time_size):
            p_model[i,:] = exp_r_del_t.dot(p_model[i-1,:])
        return p_model

    def residuals(params):
        kappa,p_zero = params[:no_kappas], params[no_kappas:]
        r_of_t = matkappa_matr(kappa)
        exp_r_del_t = expm(r_of_t*delta_t)
        p_model_result = p_model(exp_r_del_t,p_zero)
        residual = (abs(population_data -p_model_result)).flatten()
        return residual

    def p_model_split(params):
        kappa,p_zero = params[:no_kappas], params[no_kappas:]
        r_of_t = matkappa_matr(kappa)
        t_diff = t_one-time[0]
        new_p_zero = expm(r_of_t*t_diff).dot(p_zero)
        return new_p_zero

    #Inputs 
    time = data[:,0]
    delta_t = time[1]-time[0]
    no_states = len(data[0])-1
    if np.any(initial_p_zero<0):
        initial_p_zero = np.zeros_like(initial_p_zero)
    params = np.concatenate((initial_kappas,initial_p_zero))
    
    no_kappas = (no_states*(no_states-1))//2
    # population_data = data[:,1:]
    # time_size = time.size
    
    train_set = data[data[:,0] <= t_one]
    train_time = train_set[:,0]
    time_size = train_time.size
    population_data = train_set[:,1:]

    #Optimization
    least_squares_result = least_squares(residuals, params, bounds=(0,float("inf")), verbose=0, xtol=None, ftol=1e-8) # bounds=(-float("inf"),1e-17)
    optimized_kappa = least_squares_result.x[:no_kappas]
    optimized_pzero = least_squares_result.x[no_kappas:]
    optimized_params = least_squares_result.x

    r_of_t_ls = matkappa_matr(optimized_kappa)
    exp_ls_del_t = expm(r_of_t_ls*delta_t)
    test_set = data[data[:,0] >= t_one]
    test_time = test_set[:,0]
    time_size = test_time.size
    population_data = test_set[:,1:]
    
    #p_zero = p_model(exp_ls_del_t,optimized_pzero)[-1]
    #New p_0 for calculation of residuals from t_one to t_two
    p_zero = p_model_split(optimized_params)
    optimized_params = np.concatenate((optimized_kappa, p_zero))
    residual = sum(residuals(optimized_params))/time_size#/no_states # Calculate the residual between t_1 and t_2 only
    
    #full_time = full_time[full_time[:]<=t_one]
    full_time = full_time[full_time[:] >=t_one]
    full_time = full_time[:,np.newaxis] # IF we want to propogate the calculation to calculate from 0 to the full length of time that we have
    time_size = full_time.size
    optimized_population = np.array(p_model(exp_ls_del_t,p_zero))
    optimized_population = np.concatenate((full_time, optimized_population), axis=1)

    return residual, optimized_population, optimized_kappa