import numpy as np
import matplotlib.pyplot as plt
import datetime

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
    print("Implementation B")
    print("energy =", energy)
    
    return new_m

def curve_reconstruction(N, M, points, grid_potential, n_iter, save=False):
    m0 = Utils.simple_curve(N, M, points, chain=False)
    m = chambolle_pock(N, M, grid_potential, m0, n_iter, sigma=np.sqrt(np.sqrt(M*N)), tau=.99/np.sqrt(np.sqrt(M*N)))
    plt.imshow(np.hypot(m[:,:,0], m[:,:,1]))
    plt.show()
    plt.close()
    if save:
        plt.imsave(str(datetime.datetime.now())+"_B.pdf", np.hypot(m[:,:,0], m[:,:,1]))

def curve_discovery(N, M, points, grid_potential, n_iter, n_moves):
    m0 = Utils.simple_curve(N, M, points, chain=False)
    m = chambolle_pock(N, M, grid_potential, m0, n_iter, sigma=np.sqrt(np.sqrt(M*N)), tau=.99/np.sqrt(np.sqrt(M*N)))
    curve = np.hypot(m[:,:,0], m[:,:,1])

    square = np.stack([np.repeat(np.arange(-1,2),3).reshape((3,3)), np.transpose(np.repeat(np.arange(-1,2),3).reshape((3,3)))], axis=-1)
    
    for n in range(n_moves):
        #source
        curve_source = curve[points[1][1] + square[:,:,0], points[1][0] + square[:,:,1]]
        m_source = m[points[1][1] + square[:,:,0], points[1][0] + square[:,:,1]]
        potentials_source = grid_potential[points[1][1] + square[:,:,0], points[1][0] + square[:,:,1]] - 0.5 #shifting by 0.5
        curve_source_argmax = square[np.argmax(curve_source)//3, np.argmax(curve_source)%3]
        m_source_max = m_source[curve_source_argmax[0]+1, curve_source_argmax[1]+1]
        
        curve_direction_source = np.array([(np.abs(m_source_max[0]) > 0.3)*np.sign(m_source_max[0]), (np.abs(m_source_max[1]) > 0.3)*np.sign(m_source_max[1])], dtype="int")
        
        if potentials_source[curve_direction_source[0]+1, curve_direction_source[1]+1] > 0:
            points[0][:-1] += (points[1][:-1] + np.flip(curve_direction_source) > 0)*(points[1][:-1] + np.flip(curve_direction_source) < [M-1, N-1])*np.flip(curve_direction_source) #need to flip because of matplotlib
        elif np.transpose(potentials_source)[curve_direction_source[0]+1, curve_direction_source[1]+1] < 0:
            points[1][:-1] -= (points[1][:-1] - np.flip(curve_direction_source) > 0)*(points[1][:-1] - np.flip(curve_direction_source) < [M-1, N-1])*np.flip(curve_direction_source)

        #sink
        curve_sink = curve[points[0][1] + square[:,:,0], points[0][0] + square[:,:,1]]
        m_sink = m[points[0][1] + square[:,:,0], points[0][0] + square[:,:,1]]
        potentials_sink = grid_potential[points[0][1] + square[:,:,0], points[0][0] + square[:,:,1]] - 0.5
        curve_sink_argmax = square[np.argmax(curve_sink)//3, np.argmax(curve_sink)%3]
        m_sink_max = m_sink[curve_sink_argmax[0]+1, curve_sink_argmax[1]+1]
        
        curve_direction_sink = -np.array([(np.abs(m_sink_max[0]) > 0.3)*np.sign(m_sink_max[0]), (np.abs(m_sink_max[1]) > 0.3)*np.sign(m_sink_max[1])], dtype="int") #minus sign because of sink
        
        if potentials_sink[curve_direction_sink[0]+1, curve_direction_sink[1]+1] > 0:
            points[0][:-1] += (points[0][:-1] + np.flip(curve_direction_sink) > 0)*(points[0][:-1] + np.flip(curve_direction_sink) < [M-1, N-1])*np.flip(curve_direction_sink)
        elif np.transpose(potentials_sink)[curve_direction_sink[0]+1, curve_direction_sink[1]+1] < 0:
            points[0][:-1] -= (points[0][:-1] - np.flip(curve_direction_sink) > 0)*(points[0][:-1] - np.flip(curve_direction_sink) < [M-1, N-1])*np.flip(curve_direction_sink)
        
        m0 = Utils.simple_curve(N, M, points, chain=False)
        m = chambolle_pock(N, M, grid_potential, m0, n_iter, sigma=np.sqrt(np.sqrt(M*N)), tau=.99/np.sqrt(np.sqrt(M*N)))
        curve = np.hypot(m[:,:,0], m[:,:,1])
        plt.imshow(curve)
        plt.show()
        plt.close()
