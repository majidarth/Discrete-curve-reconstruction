import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

import Potential_initialization
import Utils

def L2_inv(N, M, z):
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

def chambolle_pock_projection(N, M, grid_potential, m0, n_iter, sigma, tau, theta=1):
    old_m = m0.copy()
    new_m = m0.copy()
    m_barre = m0.copy()
    z = np.zeros((M-1,N-1,2))

    for n in range(n_iter):
        
        z += sigma*m_barre
        
        # projecting for L1
        Filt = 0.25*np.ones((2,2))
        L1_z_x = scp.signal.convolve2d(np.pad(z[:,:,0], [(1,0), (0,1)], mode="symmetric"), Filt, mode="valid")
        L1_z = np.stack([L1_z_x, z[:,:,1]], axis=-1)
        mask = np.hypot(L1_z[:,:,0], L1_z[:,:,1]) > grid_potential

        L1_z[mask] *= (grid_potential[mask]/np.hypot(L1_z[mask,0], L1_z[mask,1]))[:,None]

        z[:,:,0] = np.transpose(L2_inv(M, N, np.transpose(L1_z[:,:,0])))
        z[:,:,1] = L1_z[:,:,1]
        
        # projecting for L2
        L2_z_y = scp.signal.convolve2d(np.pad(z[:,:,1], [(0,1), (1,0)], mode="symmetric"), Filt, mode="valid")
        L2_z = np.stack([z[:,:,0], L2_z_y], axis=-1)
        mask = np.hypot(L2_z[:,:,0], L2_z[:,:,1]) > grid_potential

        L2_z[mask] *= (grid_potential[mask]/np.hypot(L2_z[mask,0], L2_z[mask,1]))[:,None]

        z[:,:,0] = L2_z[:,:,0]
        z[:,:,1] = L2_inv(N, M, L2_z[:,:,1])

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

def chambolle_pock_dualvar(N, M, grid_potential, m0, n_iter, sigma, tau, theta=1):
    old_m = m0.copy()
    old_q = np.zeros((3,M-1,N-1,2))
    new_m = m0.copy()
    new_q = np.zeros((3,M-1,N-1,2))
    m_barre = m0.copy()
    q_barre = np.zeros((3,M-1,N-1,2))
    z = np.zeros((M-1,N-1,2))

    for n in range(n_iter):

        z += sigma*m_barre
        #L1*
        Filt = 0.25*np.ones((2,2))
        z[:,:,0] -= sigma*scp.signal.convolve2d(np.pad(q_barre[0,:,:,0], [(0,1), (1,0)], mode="symmetric"), Filt, mode="valid")
        z[:,:,1] -= sigma*q_barre[0,:,:,1]
        #L2*
        z[:,:,0] -= sigma*q_barre[1,:,:,0]
        z[:,:,1] -= sigma*scp.signal.convolve2d(np.pad(q_barre[1,:,:,1], [(1,0), (0,1)], mode="symmetric"), Filt, mode="valid")
        #L3*
        L3a_qbarre_x = np.vstack([0.5*q_barre[2,:-1,:,0] + 0.5*q_barre[2,1:,:,0], q_barre[2,-1,:,0]])
        L3a_qbarre_y = np.hstack([0.5*q_barre[2,:,:-1,1] + 0.5*q_barre[2,:,1:,1], np.reshape(q_barre[2,:,-1,0], (M-1,1))])
        L3a_qbarre = np.stack([L3d_qbarre_x, L3d_qbarre_y], axis=-1)
        z -= sigma*L3a_qbarre
        
        old_m = new_m.copy()
        new_m -= tau*z
        # projecting
        new_m = m0 + Utils.proj_div(N, M, new_m - m0, chain=False)
        m_barre = new_m + theta*(new_m - old_m)

        old_q = new_q.copy()
        #L1
        new_q[0,:,:,0] += tau*scp.signal.convolve2d(np.pad(z[:,:,0], [(1,0), (0,1)], mode="symmetric"), Filt, mode="valid")
        new_q[0,:,:,1] += tau*z[:,:,1]
        #L2
        new_q[1,:,:,0] += tau*z[:,:,0]
        new_q[1,:,:,1] += tau*scp.signal.convolve2d(np.pad(z[:,:,1], [(0,1), (1,0)], mode="symmetric"), Filt, mode="valid")
        #L3
        L3_z_x = np.vstack([z[0,:,0], 0.5*z[1:,:,0] + 0.5*z[:-1,:,0]])
        L3_z_y = np.hstack([np.reshape(z[:,0,1], (M-1,1)), 0.5*z[:,1:,1] + 0.5*z[:,:-1,1]])
        L3_z = np.stack([L3_z_x, L3_z_y], axis=-1)

        new_q[2,:,:,:] += tau*L3_z
        
        mask = np.hypot(new_q[:,:,:,0], new_q[:,:,:,1]) > tau
        grid_potential_bc = np.broadcast_to(grid_potential, (3,N-1,M-1))
        new_q[mask] -= tau*new_q[mask]*(grid_potential_bc[mask]/(np.hypot(new_q[mask,0], new_q[mask,1])))[:,None]
        new_q[1-mask] = 0
        q_barre = new_q + theta*(new_q - old_q)
        
    energy = np.sum(new_m*z)
    print("energy = ", energy)
    
    return new_m
    
def curve_reconstruction(N, M, points, grid_potential, n_iter):
    m0 = Utils.simple_curve(N, M, points, chain=False)
    m = chambolle_pock_dualvar(N, M, grid_potential, m0, n_iter, sigma=.99/np.sqrt(np.sqrt(M*N)), tau=np.sqrt(np.sqrt(M*N)))
    plt.imshow(np.hypot(m[:,:,0], m[:,:,1]))
    plt.show()
    plt.close()
