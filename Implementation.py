import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import datetime

import Utils

def chambolle_pock(N, M, grid_potential, m1, z1, m0, n_iter, sigma, tau, theta=1):
    old_m = m1.copy()
    new_m = m1.copy()
    m_barre = m1.copy()
    z = z1

    for n in range(n_iter):
        z += sigma*np.stack([np.vstack([0.5*m_barre[0,:,0], 0.5*m_barre[:-1,:,0] + 0.5*m_barre[1:,:,0]]), np.hstack([np.reshape(0.5*m_barre[:,0,1], (M-1,1)), 0.5*m_barre[:,:-1,1] + 0.5*m_barre[:,1:,1]])], axis=-1)
        # projecting
        mask = np.hypot(z[:,:,0], z[:,:,1]) > grid_potential
        z[mask] *= (grid_potential[mask]/np.hypot(z[mask,0], z[mask,1]))[:,None]
        
        old_m = new_m.copy()
        new_m -= tau*np.stack([np.vstack([0.5*z[:-1,:,0] + 0.5*z[1:,:,0], 0.5*z[-1,:,0]]), np.hstack([0.5*z[:,:-1,1] + 0.5*z[:,1:,1], np.reshape(0.5*z[:,-1,1], (M-1,1))])], axis=-1)
        # projecting
        new_m = m0 + Utils.proj_div(N, M, new_m - m0)
        m_barre = new_m + theta*(new_m - old_m)
        
    energy = np.sum(new_m*z)
    # print("energy =", energy)
    
    return new_m, z

def curve_reconstruction(N, M, points, grid_potential, n_iter, save=False):
    m0 = Utils.simple_curve(N, M, points)
    m, _ = chambolle_pock(N, M, grid_potential, m0, np.zeros((M-1,N-1,2)), m0, n_iter, sigma=np.sqrt(np.sqrt(M*N)), tau=.99/np.sqrt(np.sqrt(M*N)))
    
    plt.imshow(np.hypot(m[:,:,0], m[:,:,1]))
    plt.show()
    plt.close()
    if save:
        plt.imsave(str(datetime.datetime.now())+".pdf", np.hypot(m[:,:,0], m[:,:,1]))

def curve_discovery(N, M, points, grid_potential, n_iter, n_moves, max_pot, min_dir=0.05):
    saved_curve = np.zeros((M-1,N-1,n_moves))
    square = np.stack([np.repeat(np.arange(-1,2),3).reshape((3,3)), np.transpose(np.repeat(np.arange(-1,2),3).reshape((3,3)))], axis=-1)
    rot = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
    previous_steps = [[] for i in range(len(points))]

    m = Utils.simple_curve(N, M, points)
    z = np.zeros((M-1,N-1,2))
    
    for n in range(n_moves):
        m0 = Utils.simple_curve(N, M, points)
        m, z = chambolle_pock(N, M, grid_potential, m, z, m0, n_iter, sigma=np.sqrt(np.sqrt(M*N)), tau=.99/np.sqrt(np.sqrt(M*N)))
        curve = np.hypot(m[:,:,0], m[:,:,1])
        saved_curve[:,:,n] = curve

        i = n%2
        while i < len(points):
            m_local = m[points[i][1] + square[:,:,0], points[i][0] + square[:,:,1]]
            potentials_local = grid_potential[points[i][1] + square[:,:,0], points[i][0] + square[:,:,1]]
            m_mean = np.mean(m_local, axis=(0,1))
            curve_direction = (-1)**(1-points[i][-1])*np.array([(np.abs(m_mean[0]) > min_dir)*np.sign(m_mean[0]), (np.abs(m_mean[1]) > min_dir)*np.sign(m_mean[1])], dtype="int") #minus sign for sink

            directions = -np.rint([curve_direction, rot@curve_direction, np.transpose(rot)@curve_direction]).astype("int") #opposite of the curve direction with its two close directions
            orth_dirs = np.rint([rot@rot@curve_direction, -rot@rot@curve_direction]).astype("int") #two orthogonal directions
            back_dirs = np.rint([rot@curve_direction, np.transpose(rot)@curve_direction]).astype("int") #two backwards directions
            
            if previous_steps[i] != []:
                mask_dir = np.prod(np.sum((points[i][:-1] + np.flip(directions, axis=1))[:,None] != np.array(previous_steps[i])[None,:], axis=2), axis=1, dtype="bool")
                directions = directions[mask_dir]

                mask_dir_orth = np.prod(np.sum((points[i][:-1] + np.flip(orth_dirs, axis=1))[:,None] != np.array(previous_steps[i])[None,:], axis=2), axis=1, dtype="bool")
                orth_dirs = orth_dirs[mask_dir_orth]

                mask_dir_back = np.prod(np.sum((points[i][:-1] + np.flip(back_dirs, axis=1))[:,None] != np.array(previous_steps[i])[None,:], axis=2), axis=1, dtype="bool")
                back_dirs = back_dirs[mask_dir_back]

            directions_potentials = potentials_local[directions[:,0]+1, directions[:,1]+1]
            orth_potentials = potentials_local[orth_dirs[:,0]+1, orth_dirs[:,1]+1]
            back_potentials = potentials_local[back_dirs[:,0]+1, back_dirs[:,1]+1]

            if potentials_local[1,1] >= max_pot:
                vector = np.flip(curve_direction) #need to flip because of matplotlib
                points[i][:-1] += (points[i][:-1] + vector > 0)*(points[i][:-1] + vector < [N-2, M-2])*vector

            elif len(directions) != 0 and np.min(directions_potentials) < max_pot:
                vector = np.flip(directions[np.argmin(directions_potentials)])
                previous_steps[i].append(points[i][:-1])
                points[i][:-1] += (points[i][:-1] + vector > 0)*(points[i][:-1] + vector < [N-2, M-2])*vector

            elif len(orth_dirs) != 0 and np.min(orth_potentials) < max_pot:
                vector = np.flip(orth_dirs[np.argmin(orth_potentials)])
                previous_steps[i].append(points[i][:-1])
                points[i][:-1] += (points[i][:-1] + vector > 0)*(points[i][:-1] + vector < [N-2, M-2])*vector

            elif len(back_dirs) != 0 and np.min(back_potentials) < max_pot:
                vector = np.flip(back_dirs[np.argmin(back_potentials)])
                previous_steps[i].append(points[i][:-1])
                points[i][:-1] += (points[i][:-1] + vector > 0)*(points[i][:-1] + vector < [N-2, M-2])*vector

            #discarding neighbouring Dirac masses with different signs            
            if np.sum(np.prod(np.abs(points[i][:-1] - np.array(points)[i+1:,:-1]) < 2, axis=1)*(np.array(points)[i+1:,-1] != np.array(points)[i,-1])):
                points.pop(int(i+1+np.argwhere(np.prod(np.abs(points[i][:-1] - np.array(points)[i+1:,:-1]) < 2, axis=1)*(np.array(points)[i+1:,-1] != np.array(points)[i,-1]))[0]))
                points.pop(i)
                i -= 1

            i += 2

        # if n%50 == 0:
        #     # plt.imsave("curve_"+str(n)+".pdf",curve)
        #     # print(n, len(points))
        #     plt.imshow(curve)
        #     plt.show()
        #     plt.close()
    
    # fig,ax = plt.subplots()
    # def animate_curve(i):
    #     ax.clear()
    #     plot = ax.imshow(saved_curve[:,:,i])
    #     return plot

    # animation = anim.FuncAnimation(fig, animate_curve, repeat=True, frames=n_moves)
    # animation.save(str(datetime.datetime.now())+"_curve.mp4")
    
    m0 = Utils.simple_curve(N, M, points)
    m, _ = chambolle_pock(N, M, grid_potential, m, z, m0, n_iter, sigma=np.sqrt(np.sqrt(M*N)), tau=.99/np.sqrt(np.sqrt(M*N)))

    threshold_1 = 1e-3
    threshold_2 = 1e-2
    dirac_pairs, integrated_curve = Utils.integrate_curves(N, M, m, points, threshold_1, threshold_2)
    # print(dirac_pairs)
    new_points = Utils.postprocess_curves(N, M, dirac_pairs, integrated_curve)
    # print(points)
    # print(new_points)
    curve_reconstruction(N, M, new_points, grid_potential, n_iter*100, save=True)
