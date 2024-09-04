import numpy as np
import matplotlib.pyplot as plt

import Potential_initialization
import Utils

def chambolle_pock(N, M, g, m0, n_iter, sigma, tau, theta=1):
    n_nodes = N*M
    n_edges = N*(M-1) + M*(N-1)
    
    old_m = m0.copy()
    new_m = m0.copy()
    m_barre = m0.copy()
    u = np.zeros(n_nodes)
    
    for n in range(n_iter):
        # updating u
        # horizontal edges
        u[np.arange(n_nodes)[np.arange(n_nodes)%N != N-1]] -= sigma*(m_barre - m0)[np.arange(M*(N-1))]
        u[np.arange(n_nodes)[np.arange(n_nodes)%N != 0]] += sigma*(m_barre - m0)[np.arange(M*(N-1))]
        # vertical edges
        vertical_numbering = np.resize(np.arange(N)[:,None] + N*np.arange(M)[None,:], n_nodes)
        u[vertical_numbering[vertical_numbering >= N]] += sigma*(m_barre - m0)[M*(N-1) + np.arange(N*(M-1))]
        u[vertical_numbering[vertical_numbering < N*(M-1)]] -= sigma*(m_barre - m0)[M*(N-1) + np.arange(N*(M-1))]

        # updating m
        shrink = np.zeros(n_edges)
        shrink = new_m.copy()

        # horizontal edges
        shrink[np.arange(M*(N-1))] += tau*u[np.arange(n_nodes)[np.arange(n_nodes)%N != N-1]]
        shrink[np.arange(M*(N-1))] -= tau*u[np.arange(n_nodes)[np.arange(n_nodes)%N != 0]]

        # vertical edges
        shrink[M*(N-1) + np.arange(N*(M-1))] -= tau*u[vertical_numbering[vertical_numbering >= N]]
        shrink[M*(N-1) + np.arange(N*(M-1))] += tau*u[vertical_numbering[vertical_numbering < N*(M-1)]]
        
        old_m = new_m.copy()
        # new_m = (shrink > tau*g + alpha)*(shrink - tau*g - alpha) + (shrink < -tau*g + alpha)*(shrink + tau*g - alpha) #random linear functional alpha
        new_m = (shrink > tau*g)*(shrink - tau*g) + (shrink < -tau*g)*(shrink + tau*g)
        
        m_barre = new_m + theta*(new_m - old_m)

    energy = np.sum(g*np.abs(new_m))
    print("energy =", energy)
        
    return new_m

def curve_reconstruction(N, M, points, grid_potential, n_iter):
    m0 = Utils.simple_curve(N, M, points)
    
    g = Potential_initialization.edge_potential(N, M, grid_potential)
    m = chambolle_pock(N, M, g, m0, n_iter, sigma=np.sqrt(np.sqrt(M*N)), tau=.99/(8*np.sqrt(np.sqrt(M*N))))
    z = Utils.chain_to_vf(N, M, m)
    plt.imshow(np.hypot(z[:,:,0], z[:,:,1]))
    plt.show()
    plt.close()
    # Utils.plot_chain(N, M, m, grid_potential=grid_potential, plot_potential=True)
    # Utils.plot_chain(N, M, m, plot_all_edges=False)
