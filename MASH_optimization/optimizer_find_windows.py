import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

def optimize(data, initial_kappas, initial_p_zero, eq_pop, full_time, t_one,weight): #t_one, t_two, full_time): #full_time, full_population

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
        residual = (population_data-p_model_result).flatten()
        additional_resid = eq_pop-p_model_split(params,full_time[-1])
        residual = np.append(residual,additional_resid*weight)
        return residual

    def p_model_split(params,t):
        kappa,p_zero = params[:no_kappas], params[no_kappas:]
        r_of_t = matkappa_matr(kappa)
        t_diff = t-time[0]
        new_p_zero = expm(r_of_t*t_diff).dot(p_zero)
        return new_p_zero

    #Inputs 
    time = data[:,0]
    delta_t = time[1]-time[0]
    no_states = len(data[0])-1
    # INCLUSION OF P_0 INTO THE OPTIMIZATION PARAMETERS
    if np.any(initial_p_zero<0):
        initial_p_zero = np.zeros_like(initial_p_zero)
        initial_p_zero[0] = 1
    params = np.concatenate((initial_kappas,initial_p_zero))
    # params = initial_kappas
    
    no_kappas = (no_states*(no_states-1))//2
    # population_data = data[:,1:]
    # time_size = time.size
    
    train_set = data[data[:,0] <= t_one]
    train_time = train_set[:,0]
    time_size = train_time.size
    population_data = train_set[:,1:]

    #BOUNDARIES FOR IF WE ARE OPTIMIZING FOR P_0
    kappa_bounds = [(0, float("inf"))]*no_kappas
    pop_bounds = [(0,1)]*no_states
    tot_bounds = kappa_bounds+pop_bounds
    lower_bounds, upper_bounds = zip(*tot_bounds)
    tot_bounds = (list(lower_bounds), list(upper_bounds))
    
    p_zero = initial_p_zero
    least_squares_result = least_squares(residuals, params, bounds=tot_bounds, verbose=0, xtol=None, ftol=1e-8) # bounds=(-float("inf"),1e-17)
    optimized_kappa = least_squares_result.x[:no_kappas]
    optimized_pzero = least_squares_result.x[no_kappas:]
    optimized_params = least_squares_result.x
    # optimized_kappa = least_squares_result.x

    r_of_t_ls = matkappa_matr(optimized_kappa)
    t_diff = full_time[-1]
    eq_pop_calc = expm(r_of_t_ls*t_diff).dot(optimized_pzero)
    exp_ls_del_t = expm(r_of_t_ls*delta_t)
    neg_exp_ls_del_t = expm(r_of_t_ls*-delta_t)
    
    test_set = data[data[:,0] >= t_one]
    test_time = test_set[:,0]
    time_size = test_time.size
    population_data = test_set[:,1:]
    #p_zero = p_model(exp_ls_del_t,optimized_pzero)[-1]
    #New p_0 for calculation of residuals from t_one to t_two
    
    p_zero = p_model_split(optimized_params,t_one)
    optimized_params = np.concatenate((optimized_kappa, p_zero))

    additional_resid = eq_pop-p_model_split(optimized_params,full_time[-1])
    #residual = (sum(abs(residuals(optimized_params)))+sum(abs(additional_resid)))/time_size#/no_states # Calculate the residual between t_1 and t_2 only
    residual = residuals(optimized_params)
    residual_1 = ((sum(abs(residual))-sum(abs(additional_resid)))/time_size+weight)/no_states
    residual_2 = (sum(abs(residual))/time_size)/no_states
    #residual_3 = (sum(abs(residual))/time_size)/no_states

    # #full_time = full_time[full_time[:]<=t_one]
    # full_time = full_time[full_time[:] >=t_one]
    # IF we want to propogate the calculation to calculate from 0 to the full length of time that we have
    # if time[0] != 0.0:
    #     post_time = full_time[full_time[:]>=time[0]]
    #     time_size = post_time.size
    #     optimized_population_fromtzero = np.array(p_model(exp_ls_del_t,optimized_pzero))
    #     pre_time = full_time[full_time[:]< time[0]]
    #     print("hello")
    #     time_size = pre_time.size
    #     optimized_population_beforetzero = np.array(p_model(neg_exp_ls_del_t, optimized_pzero))
    #     optimized_population_beforetzero = optimized_population_beforetzero[::-1]
    #     print(optimized_population_beforetzero.shape)
    #     optimized_population = np.vstack((optimized_population_beforetzero,optimized_population_fromtzero))
    #     print(optimized_population.shape)
    # else:
    #     time_size = full_time.size
    #     optimized_population = np.array(p_model(exp_ls_del_t,optimized_pzero))

    #UNCOMMENT
    full_time = full_time[full_time[:]>=t_one]
    time_size = full_time.size
    delta_t = full_time[1]-full_time[0]
    exp_ls_del_t = expm(r_of_t_ls*delta_t)
    optimized_population = np.array(p_model(exp_ls_del_t,p_zero)) 
    full_time = full_time[:,np.newaxis] 
    optimized_population = np.concatenate((full_time, optimized_population), axis=1)

    
    return residual_1,residual_2, optimized_population, optimized_kappa, eq_pop_calc
    #return optimized_kappa, eq_pop_calc