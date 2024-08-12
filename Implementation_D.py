import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

import Potential_initialization
import Utils

def L1_inv(N, M, z):
    for i in range(1,M-1):
        z[M-2-i,0] = 2*z[M-2-i,0] - z[M-2-(i-1),0]
        
    for j in range(1,N-1):
        z[M-2,j] = 2*z[M-2,j] - z[M-2,j-1]
        
    for k in range(1, min(N-1, M-1)):
        z[M-2-k,k] = 4*z[M-2-k,k] - z[M-2-(k-1),k] - z[M-2-k,k-1] - z[M-2-(k-1),k-1]
        for i in range(k+1,M-1):
            z[M-2-i,k] = 4*z[M-2-i,k] - z[M-2-(i-1),k] - z[M-2-i,k-1] - z[M-2-(i-1),k-1]
        for j in range(k+1,N-1):
            z[M-2-k,j] = 4*z[M-2-k,j] - z[M-2-(k-1),j] - z[M-2-k,j-1] - z[M-2-(k-1),j-1]

    return z


def L3_inv(N, M, z):
    for k in range(1, min(N-1,M-1)):
        z[k,:,0] = 2*z[k,:,0] - z[k-1,:,0]
        z[:,k,1] = 2*z[:,k,1] - z[:,k-1,1]
    if M > N:
        for k in range(N-1, M-1):
            z[k,:,0] = 2*z[k,:,0] - z[k-1,:,0]
    else:
        for k in range(M-1, N-1):
            z[:,k,1] = 2*z[:,k,1] - z[:,k-1,1]

    return z

def chambolle_pock(N, M, grid_potential, m0, n_iter, sigma, tau, theta=1):
    old_m = m0.copy()
    new_m = m0.copy()
    m_barre = m0.copy()
    z = np.zeros((M-1,N-1,2))

    for n in range(n_iter):

        if n%1000 == 0:
            print(n)
        
        z += sigma*m_barre
        
        # projecting for L1
        Filt = 0.25*np.ones((2,2))
        L1_z_x = scp.signal.convolve2d(np.pad(z[:,:,0], [(0,1), (1,0)], mode="symmetric"), Filt, mode="valid")
        L1_z = np.stack([L1_z_x, z[:,:,1]], axis=-1)
        mask = np.hypot(L1_z[:,:,0], L1_z[:,:,1]) > grid_potential

        L1_z[mask] *= (grid_potential[mask]/np.hypot(L1_z[mask,0], L1_z[mask,1]))[:,None]

        z[:,:,0] = L1_inv(N, M, L1_z[:,:,0])
        z[:,:,1] = L1_z[:,:,1]
        
        # projecting for L2
        L2_z_y = scp.signal.convolve2d(np.pad(z[:,:,1], [(1,0), (0,1)], mode="symmetric"), Filt, mode="valid")
        L2_z = np.stack([z[:,:,0], L2_z_y], axis=-1)
        mask = np.hypot(L2_z[:,:,0], L2_z[:,:,1]) > grid_potential

        L2_z[mask] *= (grid_potential[mask]/np.hypot(L2_z[mask,0], L2_z[mask,1]))[:,None]

        z[:,:,0] = L2_z[:,:,0]
        z[:,:,1] = np.transpose(L1_inv(M, N, np.transpose(L2_z[:,:,1])))

        # projecting for L3
        L3_z_x = np.vstack([z[0,:,0], 0.5*z[1:,:,0] + 0.5*z[:-1,:,0]])
        L3_z_y = np.hstack([np.reshape(z[:,0,1], (M-1,1)), 0.5*z[:,1:,1] + 0.5*z[:,:-1,1]])
        L3_z = np.stack([L3_z_x, L3_z_y], axis=-1)
        mask = np.hypot(L3_z[:,:,0], L3_z[:,:,1]) > grid_potential

        L3_z[mask] *= (grid_potential[mask]/np.hypot(L3_z[mask,0], L3_z[mask,1]))[:,None]

        z = L3_inv(N, M, L3_z)
        
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
