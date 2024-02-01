import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

class Optimizer:
    def __init__(self, data, ns, ks, kt,initial_kappas=None):
        ''' Initialise the optimizer.
        data            data to be fit, including time columns, Should I assert that the column with 1 is the singlet state?
        ns              number of states
        initial_kappas  initial kappas (if unsure, leave None)
        kt, ks          rate constant of the triplet and singlet state respectively in mus
        '''
        assert data.shape[1] == ns+1
        if(initial_kappas is None):
            self.initial_kappas = np.zeros(ns*(ns-1)//2)
        else:
          assert len(initial_kappas)==(ns*(ns-1)/2)
          self.initial_kappas = initial_kappas

        data = data[::-1,:]
        self.time = data[:,0]
        self.pop = data[:,1:5]
#       self.eq_pop = eq_pop
        self.ns = ns
        self.time_size = len(self.time)
        self.dt = self.time[1]-self.time[0] #This gives a negative value for going backwards
        self.p0 = self.pop[0]
        self.kappa_opt = None
        self.ks = ks
        self.kt = kt
        self.known_k = [self.ks]+[self.kt]*(self.ns-1)

    # def listkap_matkappa(self, kappa):
    #     ''' Calculate the kappa matrix from the list of kappas.
    #     '''
    #     matkappa = np.zeros((self.ns,self.ns))
    #     triangle = np.triu(np.ones((self.ns, self.ns), dtype=bool), k=1)
    #     matkappa[triangle]= kappa
    #     matkappa += matkappa.T
    #     return matkappa
        
    def test(self):
        a = self.known_k
        return a

    def listkap_matkappa(self, kappa):
        ''' Calculate the kappa matrix from the list of diagonal and upper right non-diagonal kappas.
        '''
        matkappa = np.zeros((self.ns,self.ns))
        triangle = np.triu(np.ones((self.ns, self.ns), dtype=bool), k=1)
        matkappa[triangle] = -kappa  # Populate upper right non-diagonal elements
        matkappa += matkappa.T  # Symmetric adjustment
        for i in range(self.ns): 
            matkappa[i, i] = self.known_k[i] + np.sum(matkappa[i, :])

        return matkappa

    def p_model(self, exp_r_del_t):
        ''' Evaluate the training model.
        '''
        p_model = np.zeros_like(self.pop)
        p_model[0,:] = self.p0
        for i in range(1,self.time_size):
            p_model[i,:] = exp_r_del_t.dot(p_model[i-1,:])
        return p_model

    def residuals(self, kappa):
        ''' Calculate the vector of residuals.
        '''
        kappa_of_t = self.listkap_matkappa(kappa)
        exp_r_del_t = expm(kappa_of_t*self.dt)
        p_model_result = self.p_model(exp_r_del_t)
        residual = (self.pop - p_model_result).flatten()
        return residual

    def run(self):
        ''' Run an optimization.
        Returns
        res       root mean square error
        kappa_opt optimal kappa
        '''
        ls_result = least_squares(self.residuals,self.initial_kappas,bounds=(0,float("inf")),xtol=None,ftol=1e-8)
        kappa_opt = ls_result.x
        self.kappa_opt = kappa_opt
        res = np.sqrt(2.*ls_result.cost/(self.time_size*self.ns))
        return res, kappa_opt

    def predict(self,p0,time,direction,kappa):
        """
        p0: Initial population at time[0], 1D array of size N
        direction: Which way to propagate
            'forward': p0 corresponds to the value at time[0]
            'back': p0 corresponds to the value at time[-1]
        time: times at which to calculate populations
        direction
        """
        if (direction == 'back'):
            time = time[::-1]
        assert len(kappa) == self.ns*(self.ns-1)//2
        matkappa = self.listkap_matkappa(kappa)
        dt = time[1]-time[0]
        exp_rdt = expm(matkappa*dt)

        pop = np.zeros((len(time),self.ns))
        pop[0,:] = p0
        for i in range(1,len(time)):
            pop[i,:] = exp_rdt.dot(pop[i-1,:])

        if (direction == 'back'):
          pop = pop[::-1,:]
        return pop
