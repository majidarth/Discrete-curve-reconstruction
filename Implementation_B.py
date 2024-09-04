import numpy as np
import matplotlib.pyplot as plt

import Utils
    
def chambolle_pock(N, M, grid_potential, m0, n_iter, sigma, tau, theta=1):
    n_faces = (N-1)*(M-1)
    
    old_m = m0.copy()
    new_m = m0.copy()
    m_barre = m0.copy()
    z = np.zeros((n_faces,2))
    
    for n in range(n_iter):
        # updating z
        # horizontal edges
        z[:,0] += sigma/2*(m_barre[:(M-1)*(N-1)] + m_barre[N-1:M*(N-1)])        
        # vertical edges
        vertical_numbering_edges = M*(N-1) + np.resize(np.arange(M-1)[:,None] + (M-1)*np.arange(N)[None,:], N*(M-1))
        z[:,1] += sigma/2*(m_barre[vertical_numbering_edges[vertical_numbering_edges < M*(N-1) + (N-1)*(M-1)]] + m_barre[vertical_numbering_edges[vertical_numbering_edges >= M*(N-1) + (M-1)]])

        # projecting
        mask = np.hypot(z[:,0], z[:,1]) > np.resize(grid_potential, n_faces)
        z[mask,:] = z[mask,:]/np.hypot(z[mask,0], z[mask,1])[:,None]*np.resize(grid_potential, n_faces)[mask,None]
        
        # updating m
        old_m = new_m.copy()
        
        # horizontal edges
        new_m[:(M-1)*(N-1)] -= tau/2*z[:,0]
        new_m[(N-1):M*(N-1)] -= tau/2*z[:,0]
        # vertical edges
        vertical_numbering_faces = np.resize(np.arange(N-1)[:,None] + (N-1)*np.arange(M-1)[None,:], n_faces)
        new_m[M*(N-1):M*(N-1) + (M-1)*(N-1)] -= tau/2*z[vertical_numbering_faces,1]
        new_m[M*(N-1) + (M-1):] -= tau/2*z[vertical_numbering_faces,1]

        #projecting
        new_m = m0 + Utils.proj_div(N, M, new_m - m0)
        
        m_barre = new_m + theta*(new_m - old_m)

    energy_grid = (new_m[:(M-1)*(N-1)]/2 + new_m[N-1:M*(N-1)]/2)*z[:,0] + (new_m[M*(N-1):M*(N-1) + (M-1)*(N-1)]/2 + new_m[M*(N-1) + (M-1):]/2)*z[:,1]
    energy = np.sum(energy_grid)
    print("energy =", energy)
        
    return new_m

def curve_reconstruction(N, M, points, grid_potential, n_iter, smooth=False):
    m0 = Utils.simple_curve(N, M, points, smooth=smooth)
    m = chambolle_pock(N, M, grid_potential, m0, n_iter, sigma=np.sqrt(np.sqrt(M*N)), tau=.99/(8*np.sqrt(np.sqrt(M*N))))
    z = Utils.chain_to_vf(N, M, m)
    plt.imshow(np.hypot(z[:,:,0], z[:,:,1]))
    plt.show()
    plt.close()
    # Utils.plot_chain(N, M, m, grid_potential=grid_potential, plot_potential=True)
    # Utils.plot_chain(N, M, m, plot_all_edges=False)
