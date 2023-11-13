"""Built-in python import"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import scipy.optimize as opt
import scipy.linalg as linalg

from pyrho import ham, heom

def main():
    nsite = 7
    nbath = 7
    basis = 'site'

    kB = 0.69352	# in cm-1 / K
    hbar = 5308.8   # in cm-1 * fs

    # System Hamiltonian in cm-1
    ham_sys = np.array([[12410, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],
                        [-87.7, 12530,  30.8,   8.2,   0.7,  11.8,   4.3],
                        [  5.5,  30.8, 12210, -53.5,  -2.2,  -9.6,   6.0],
                        [ -5.9,   8.2, -53.5, 12320, -70.7, -17.0, -63.3],
                        [  6.7,   0.7,  -2.2, -70.7, 12480,  81.1,  -1.3],
                        [-13.7,  11.8,  -9.6, -17.0,  81.1, 12630,  39.7],
                        [ -9.9,   4.3,   6.0, -63.3,  -1.3,  39.7, 12440]])
    # System part of the the system-bath interaction
    # - a list of length 'nbath'
    # - currently assumes that each term has uncorrelated bath operators
    ham_sysbath = []
    for n in range(nbath):
        ham_sysbath_n = np.zeros((nsite,nsite))
        ham_sysbath_n[n,n] = 1.0
        ham_sysbath.append( ham_sysbath_n )

    eigs, U = np.linalg.eigh(ham_sys)


    #for (K,L) in [(1,5),(2,5),(3,5),(4,5),(5,5),(6,5)]:
    for (K,L) in [(1,4),(2,4)]:
    # for (K,L) in [(0,1),(0,2),(1,2),(0,4),(0,6)]:
        print(K,L)
        # Spectral densities - a list of length 'nbath'
        lamda = 35.0
        for [tau,T] in [[50.,77]]:
            omega_c = 1.0/tau # in 1/fs
            kT = kB*T
            spec_densities = [['ohmic-lorentz', lamda, omega_c]]*nbath

            my_ham = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT, hbar=hbar)

            my_heom = heom.HEOM(my_ham, L=L, K=K)

            for init in [0]:
                # Initial reduced density matrix of the system
                rho_0 = np.zeros((nsite, nsite))
                rho_0[init,init] = 1.0
                
                if basis=='exc':
                    """ From exciton to site basis """
                    rho_0 = np.einsum('kl,lm,mn->kn', U, rho_0, U.T)
                print(T, tau, init)
                times, rhos_site = my_heom.propagate(rho_0, 0.0, 10000.0, 1.0)

                if basis=='exc':
                    rhos_site = np.einsum('kl,tlm,mn->tkn',U.T,rhos_site,U)
                out = np.column_stack([times,rhos_site.reshape(-1,nsite**2)])
                # np.savetxt('rho_Re_%s_tau-%.0f_T-%.0f_init-%d_K_%d_L_%d.dat'%(basis,tau,T,init,K,L),out.real)
                # np.savetxt('rho_Im_%s_tau-%.0f_T-%.0f_init-%d_K_%d_L_%d.dat'%(basis,tau,T,init,K,L),out.imag)

                with open('pop_%s_tau-%.0f_T-%.0f_init-%d_K_%d_L_%d.dat'
                # with open('rho_real_exc_tau-%.0f_T-%.0f_init-%d_K_%d_L_%d.dat'
                           %(basis,tau,T,init,K,L), 'w') as f:
                    for (time, rho) in zip(times, rhos_site):
                        # if basis=='exc':
                        #     rho = np.einsum('kl,lm,mn->kn', U.T, rho, U)
                        # rho_exc = rho_site
                        f.write('%0.8f '%(time))
                        for i in range(nsite):
                            f.write('%0.8f '%(rho[i,i].real))
                            # f.write('%0.8f '%(rho[i,i].imag))
                        f.write('\n')

if __name__ == '__main__':
    main()
