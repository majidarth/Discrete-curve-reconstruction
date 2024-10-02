import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import datetime

import Utils

def chambolle_pock(N, M, grid_potential, z0, n_iter, sigma, tau, theta=1):
    old_q = np.zeros((3,M-1,N-1,2))
    new_q = np.zeros((3,M-1,N-1,2))
    q_barre = np.zeros((3,M-1,N-1,2))
    u = np.zeros((M-1,N-1))

    for n in range(n_iter):

        La_qbarre = np.zeros((M-1,N-1,2))
        
        #L1*
        Filt_1 = np.zeros((3,3))
        Filt_1[:-1,1:] = 0.25*np.ones((2,2))
        La_qbarre[:,:,0] += scp.signal.correlate2d(q_barre[0,:,:,0], Filt_1, mode="same")
        La_qbarre[:,:,1] += q_barre[0,:,:,1]
        
        #L2*
        Filt_2 = np.zeros((3,3))
        Filt_2[1:,:-1] = 0.25*np.ones((2,2))
        La_qbarre[:,:,0] += q_barre[1,:,:,0]
        La_qbarre[:,:,1] += scp.signal.correlate2d(q_barre[1,:,:,1], Filt_2, mode="same")
        
        #L3*
        L3a_qbarre_x = np.vstack([0.5*q_barre[2,:-1,:,0] + 0.5*q_barre[2,1:,:,0], 0.5*q_barre[2,-1,:,0]])
        L3a_qbarre_y = np.hstack([0.5*q_barre[2,:,:-1,1] + 0.5*q_barre[2,:,1:,1], 0.5*np.reshape(q_barre[2,:,-1,1], (M-1,1))])
        L3a_qbarre = np.stack([L3a_qbarre_x, L3a_qbarre_y], axis=-1)
        La_qbarre += L3a_qbarre

        Da_La_qbarre = -(np.vstack([La_qbarre[0,:,0], La_qbarre[1:-1,:,0]-La_qbarre[0:-2,:,0], -La_qbarre[-2,:,0]]) + np.hstack([np.reshape(La_qbarre[:,0,1],(M-1,1)), La_qbarre[:,1:-1,1]-La_qbarre[:,0:-2,1], np.reshape(-La_qbarre[:,-2,1], (M-1,1))]))

        u += sigma*Da_La_qbarre
        u += sigma*(-(np.vstack([z0[0,:,0], z0[1:-1,:,0]-z0[0:-2,:,0], -z0[-2,:,0]]) + np.hstack([np.reshape(z0[:,0,1],(M-1,1)), z0[:,1:-1,1]-z0[:,0:-2,1], np.reshape(-z0[:,-2,1], (M-1,1))])))
        
        old_q = new_q.copy()
        du = np.stack([np.vstack([u[1:,:]-u[:-1,:], np.zeros((1,N-1))]), np.hstack([u[:,1:]-u[:,:-1], np.zeros((M-1,1))])], axis=-1)
        #L1
        new_q[0,:,:,0] -= tau*scp.signal.convolve2d(du[:,:,0], Filt_1, mode="same")
        new_q[0,:,:,1] -= tau*du[:,:,1]
        #L2
        new_q[1,:,:,0] -= tau*du[:,:,0]
        new_q[1,:,:,1] -= tau*scp.signal.convolve2d(du[:,:,1], Filt_2, mode="same")
        #L3
        L3_du_x = np.vstack([0.5*du[0,:,0], 0.5*du[1:,:,0] + 0.5*du[:-1,:,0]])
        L3_du_y = np.hstack([0.5*np.reshape(du[:,0,1], (M-1,1)), 0.5*du[:,1:,1] + 0.5*du[:,:-1,1]])
        L3_du = np.stack([L3_du_x, L3_du_y], axis=-1)
        new_q[2,:,:,:] -= tau*L3_du

        grid_potential_bc = np.broadcast_to(grid_potential, (3,N-1,M-1))
        mask = np.hypot(new_q[:,:,:,0], new_q[:,:,:,1]) > tau*grid_potential_bc
        new_q[mask] *= (1 - tau*grid_potential_bc[mask]/np.hypot(new_q[mask,0], new_q[mask,1]))[:,None]
        new_q[np.invert(mask)] = 0
        q_barre = new_q + theta*(new_q - old_q)
        
    energy = np.sum(z0*du)
    print("Implementation D 2")
    print("energy =", energy)

    z = np.zeros((M-1,N-1,2))
    z[:,:,0] += scp.signal.correlate2d(q_barre[0,:,:,0], Filt_1, mode="same")
    z[:,:,1] += q_barre[0,:,:,1]
    z[:,:,0] += q_barre[1,:,:,0]
    z[:,:,1] += scp.signal.correlate2d(q_barre[1,:,:,1], Filt_2, mode="same")
    z[:,:,0] += np.vstack([0.5*q_barre[2,:-1,:,0] + 0.5*q_barre[2,1:,:,0], 0.5*q_barre[2,-1,:,0]])
    z[:,:,1] += np.hstack([0.5*q_barre[2,:,:-1,1] + 0.5*q_barre[2,:,1:,1], 0.5*np.reshape(q_barre[2,:,-1,1], (M-1,1))])

    return z
    
def curve_reconstruction(N, M, points, grid_potential, n_iter, save=False):
    z0 = Utils.simple_curve(N, M, points, chain=False)
    z = chambolle_pock(N, M, grid_potential, z0, n_iter, sigma=np.sqrt(np.sqrt(M*N)), tau=.99/(32*np.sqrt(np.sqrt(M*N))))
    plt.imshow(np.hypot(z[:,:,0], z[:,:,1]))
    plt.show()
    plt.close()
    
    if save:
        plt.imsave(str(datetime.datetime.now())+"_D_2.pdf", np.hypot(z[:,:,0], z[:,:,1]))
