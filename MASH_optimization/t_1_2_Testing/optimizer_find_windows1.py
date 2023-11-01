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

    def residuals(kappa):
        #kappa,p_zero = params[:no_kappas], params[no_kappas:]
        r_of_t = matkappa_matr(kappa)
        exp_r_del_t = expm(r_of_t*delta_t)
        p_model_result = p_model(exp_r_del_t,p_zero)
        residual = (population_data-p_model_result).flatten()
        # additional_resid = eq_pop-p_model_split(params,full_time[-1])
        # residual = np.append(residual,additional_resid*weight)
        return residual


    #Inputs 
    time = data[:,0]
    delta_t = time[1]-time[0]
    no_states = len(data[0])-1

    
    no_kappas = (no_states*(no_states-1))//2
    # population_data = data[:,1:]
    # time_size = time.size
    
    train_set = data[data[:,0] <= t_one]
    train_time = train_set[:,0]
    time_size = train_time.size
    population_data = train_set[:,1:]
    p_zero = population_data[0]


   # p_zero = initial_p_zero
    least_squares_result = least_squares(residuals, initial_kappas, bounds=(0,float("inf")), verbose=0, xtol=None, ftol=1e-8) # bounds=(-float("inf"),1e-17)
    # optimized_kappa = least_squares_result.x[:no_kappas]
    # optimized_pzero = least_squares_result.x[no_kappas:]
    # optimized_params = least_squares_result.x
    optimized_kappa = least_squares_result.x

    r_of_t_ls = matkappa_matr(optimized_kappa)
    t_diff = full_time[-1]
    #eq_pop_calc = expm(r_of_t_ls*t_diff).dot(optimized_pzero)
    exp_ls_del_t = expm(r_of_t_ls*delta_t)
    neg_exp_ls_del_t = expm(r_of_t_ls*-delta_t)
    
    test_set = data[data[:,0] >= t_one]
    test_time = test_set[:,0]
    time_size = test_time.size
    population_data = test_set[:,1:]
    p_zero = population_data[0]``
    #p_zero = p_model(exp_ls_del_t,optimized_pzero)[-1]
    #New p_0 for calculation of residuals from t_one to t_two
    
    #p_zero = p_model_split(optimized_kappa,t_one)
    # optimized_params = np.concatenate((optimized_kappa, p_zero))

    # additional_resid = eq_pop-p_model_split(optimized_params,full_time[-1])
    #residual = (sum(abs(residuals(optimized_params)))+sum(abs(additional_resid)))/time_size#/no_states # Calculate the residual between t_1 and t_2 only
    residual = (sum(abs(residuals(optimized_kappa)))/time_size)/no_states

    r_of_t_ls = matkappa_matr(optimized_kappa)
    exp_ls_del_t = expm(r_of_t_ls*delta_t)
    #UNCOMMENT
    full_time = full_time[full_time[:]>=t_one]
    time_size = full_time.size
    delta_t = full_time[1]-full_time[0]
    
    optimized_population = np.array(p_model(exp_ls_del_t,p_zero)) 
    full_time = full_time[:,np.newaxis] 
    optimized_population = np.concatenate((full_time, optimized_population), axis=1)
    
    return residual, optimized_population, optimized_kappa, p_zero
    #return optimized_kappa, eq_pop_calc