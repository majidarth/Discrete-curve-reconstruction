import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import datetime

import Utils

def chambolle_pock(N, M, grid_potential, z0, n_iter, sigma, tau, theta=1):
    old_z = z0.copy()
    old_q = np.zeros((3,M-1,N-1,2))
    new_z = z0.copy()
    new_q = np.zeros((3,M-1,N-1,2))
    z_barre = z0.copy()
    q_barre = np.zeros((3,M-1,N-1,2))
    p = np.zeros((M-1,N-1,2))

    for n in range(n_iter):

        p += sigma*z_barre
        
        #L1*
        Filt_1 = np.zeros((3,3))
        Filt_1[:-1,1:] = 0.25*np.ones((2,2))
        p[:,:,0] -= sigma*scp.signal.correlate2d(q_barre[0,:,:,0], Filt_1, mode="same")
        p[:,:,1] -= sigma*q_barre[0,:,:,1]
        
        #L2*
        Filt_2 = np.zeros((3,3))
        Filt_2[1:,:-1] = 0.25*np.ones((2,2))
        p[:,:,0] -= sigma*q_barre[1,:,:,0]
        p[:,:,1] -= sigma*scp.signal.correlate2d(q_barre[1,:,:,1], Filt_2, mode="same")
        
        #L3*
        L3a_qbarre_x = np.vstack([0.5*q_barre[2,:-1,:,0] + 0.5*q_barre[2,1:,:,0], 0.5*q_barre[2,-1,:,0]])
        L3a_qbarre_y = np.hstack([0.5*q_barre[2,:,:-1,1] + 0.5*q_barre[2,:,1:,1], 0.5*np.reshape(q_barre[2,:,-1,1], (M-1,1))])
        L3a_qbarre = np.stack([L3a_qbarre_x, L3a_qbarre_y], axis=-1)
        p -= sigma*L3a_qbarre
        
        old_z = new_z.copy()
        new_z -= tau*p
        # projecting
        new_z = z0 + Utils.proj_div(N, M, new_z - z0, chain=False)
        z_barre = new_z + theta*(new_z - old_z)

        old_q = new_q.copy()
        #L1
        new_q[0,:,:,0] += tau*scp.signal.convolve2d(p[:,:,0], Filt_1, mode="same")
        new_q[0,:,:,1] += tau*p[:,:,1]
        #L2
        new_q[1,:,:,0] += tau*p[:,:,0]
        new_q[1,:,:,1] += tau*scp.signal.convolve2d(p[:,:,1], Filt_2, mode="same")
        #L3
        L3_p_x = np.vstack([0.5*p[0,:,0], 0.5*p[1:,:,0] + 0.5*p[:-1,:,0]])
        L3_p_y = np.hstack([0.5*np.reshape(p[:,0,1], (M-1,1)), 0.5*p[:,1:,1] + 0.5*p[:,:-1,1]])
        L3_p = np.stack([L3_p_x, L3_p_y], axis=-1)
        new_q[2,:,:,:] += tau*L3_p

        grid_potential_bc = np.broadcast_to(grid_potential, (3,N-1,M-1))
        mask = np.hypot(new_q[:,:,:,0], new_q[:,:,:,1]) > tau*grid_potential_bc
        new_q[mask] *= (1 - tau*grid_potential_bc[mask]/np.hypot(new_q[mask,0], new_q[mask,1]))[:,None]
        new_q[np.invert(mask)] = 0
        q_barre = new_q + theta*(new_q - old_q)
        
    energy = np.sum(new_z*p)
    print("Implementation D")
    print("energy =", energy)
    
    return new_z
    
def curve_reconstruction(N, M, points, grid_potential, n_iter, save=False):
    z0 = Utils.simple_curve(N, M, points, chain=False)
    z = chambolle_pock(N, M, grid_potential, z0, n_iter, sigma=np.sqrt(np.sqrt(M*N)), tau=.99/(16*np.sqrt(np.sqrt(M*N))))
    
    plt.imshow(np.hypot(z[:,:,0], z[:,:,1]))
    plt.show()
    plt.close()
    if save:
        plt.imsave(str(datetime.datetime.now())+"_D.pdf", np.hypot(z[:,:,0], z[:,:,1]))
