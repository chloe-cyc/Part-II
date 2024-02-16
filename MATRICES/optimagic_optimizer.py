import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

class Optimizer:
    def __init__(self, time, data, eq_pop, ns, initial_kappas=None):
        ''' Initialise the optimizer.
        times           times for data to be fit
        data            data to be fit
        eq_pop          equilibrium populations
        ns              number of states
        initial_kappas  initial kappas (if unsure, leave None)
        '''
        assert len(eq_pop) == ns
        assert data.shape[1] == ns #removed ns+1 time no longer attached to time
        if(initial_kappas is None):
            self.initial_kappas = np.zeros(ns*(ns-1)//2)
        else:
          assert len(initial_kappas)==ns*(ns-1)/2
          self.initial_kappas = initial_kappas

        data = data[::-1] # Changed from data = data[::-1,:]
        self.pop = data
        self.time = time[::-1]
        self.eq_pop = eq_pop
        self.ns = ns
        self.time_size = len(self.time)

        self.dt = self.time[1]-self.time[0] #This gives a negative value for going backwards
        self.p0 = self.pop[0]
        self.kappa_opt = None

    def listkap_matkappa(self, kappa):
        ''' Calculate the kappa matrix from the list of kappas.
        '''
        matkappa = np.zeros((self.ns,self.ns))
        triangle = np.triu(np.ones((self.ns, self.ns), dtype=bool), k=1)
        matkappa[triangle]= kappa
        matkappa += matkappa.T
        return matkappa

    def matkappa_matr(self, kappa):
        ''' Calculate population propagator matrix R from a list of
        kappas.
        '''
        matkappa = self.listkap_matkappa(kappa)
        matr = np.zeros((self.ns,self.ns))
        matr = matkappa * self.eq_pop[:, np.newaxis]
        rdiag = -np.sum(matr,0)
        matr = matr + np.diag(rdiag)
        return matr

    def p_model(self, exp_r_del_t,s):
        ''' Evaluate the training model.
        '''
        p_model = np.zeros((self.time_size,self.ns)) #create a new array filled with zeroes in the same shape as self.pop 7x7 array
        p_model[0,:] = self.p0[:,s] #The first 7x7 array
        for i in range(1,self.time_size):
            p_model[i,:] = np.dot(exp_r_del_t,p_model[i-1,:])
        return p_model

    def residuals(self, kappa):
        ''' Calculate the vector of residuals.
        '''
        r_of_t = self.matkappa_matr(kappa)
        exp_r_del_t = expm(r_of_t*self.dt)
        residual_total = np.zeros((self.ns,self.ns*self.time_size))

        for s in range(self.ns):
            population = self.pop[:, :, s]
            p_model_result = self.p_model(exp_r_del_t,s)
            residual = (population - p_model_result).flatten()
            residual_total[s] = residual
        residual_total = residual_total.flatten()
        return residual_total

    def run(self):
        ''' Run an optimization.
        Returns
        res       root mean square error
        kappa_opt optimal kappa
        '''
        ls_result = least_squares(self.residuals,self.initial_kappas,bounds=(0,float("inf")))#,xtol=None,ftol=1e-8)
        kappa_opt = ls_result.x
        # self.kappa_opt = kappa_opt
        res = np.sqrt(2.*ls_result.cost/(self.time_size*self.ns)) #TRY BY HAND
        return res, kappa_opt

    def predict(self,p0,time,direction,kappa):
        """
        p0: Initial population at time[0], 2d Array of nxn
        direction: Which way to propagate
            'forward': p0 corresponds to the value at time[0]
            'back': p0 corresponds to the value at time[-1]
        time: times at which to calculate populations
        direction
        """
        if (direction == 'back'):
            time = time[::-1]
        assert len(kappa) == self.ns*(self.ns-1)//2

        pop = np.zeros((len(time),self.ns,self.ns))
        r = self.matkappa_matr(kappa)
        dt = time[1]-time[0]
        exp_rdt = expm(r*dt)
        for s in range(self.ns):
            pop_s = np.zeros((len(time),self.ns))
            pop_s[0,:] = p0[:,s]
            for i in range(1,len(time)):
                pop_s[i,:] = exp_rdt.dot(pop_s[i-1,:])
            pop[:,:,s] = pop_s[:,:]

        if (direction == 'back'):
          pop = pop[::-1]
        return pop 
