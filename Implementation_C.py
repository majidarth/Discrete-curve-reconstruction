import numpy as np
import matplotlib.pyplot as plt

import Potential_initialization
import Utils

def chambolle_pock(N, M, grid_potential, m0, n_iter, sigma, tau, theta=1):
    old_m = m0.copy()
    new_m = m0.copy()
    m_barre = m0.copy()
    z = np.zeros((M-1,N-1,2))

    for n in range(n_iter):
        z += sigma*m_barre
        # projecting
        mask = np.hypot(z[:,:,0], z[:,:,1]) > grid_potential
        z[mask] *= (grid_potential[mask]/np.hypot(z[mask,0], z[mask,1]))[:,None]
        
        old_m = new_m.copy()
        new_m -= tau*z
        # projecting
        new_m = m0 + Utils.proj_div(N, M, new_m - m0, chain=False)
        m_barre = new_m + theta*(new_m - old_m)

    energy = np.sum(new_m*z)
    print("energy = ", energy)
    
    return new_m

def curve_reconstruction(N, M, points, grid_potential, n_iter):
    m0 = Utils.simple_curve(N, M, points, chain=False)
    m = chambolle_pock(N, M, grid_potential, m0, n_iter, sigma=.99/np.sqrt(np.sqrt(M*N)), tau=np.sqrt(np.sqrt(M*N)))
    plt.imshow(np.hypot(m[:,:,0], m[:,:,1]))
    plt.show()
    plt.close()
