import numpy as np
import scipy.fftpack as sft
import matplotlib.pyplot as plt
import time
from matplotlib.backend_bases import MouseButton

import Potential_initialization
import Implementation

def click_points(event, points):
    # first click for sink, second click for source
    if event.inaxes:
        points.append([round(event.xdata + 0.5), round(event.ydata + 0.5), len(points)%2 == 0]) # translate by +0.5 because of imshow
    else:
        print("Point clicked was not inside of axes")

    print("Number of points clicked:", len(points))

def plot_grid(N, M, click=False, points=None, color=True):
    x, y = np.meshgrid(np.arange(-0.5, N - 0.5), np.arange(-0.5, M - 0.5)) # transposing by -0.5 to match imshow
    plt.plot(x, y, color="b" if color else "w")
    plt.plot(np.transpose(x), np.transpose(y), color="b" if color else "w")
    
    if click:
        plt.connect("button_press_event", lambda event: click_points(event, points))

def plot_chain(N, M, c, grid_potential=None, plot_all_edges=True, plot_potential=False, threshold=1e-2):
    if plot_all_edges:
        plot_grid(N, M, color=True)
    if plot_potential:
        plt.imshow(grid_potential, cmap="gray")

    vertical_edges = np.zeros((2*N*(M-1), 2))
    vertical_edges[2*np.arange(N*(M-1))] = np.arange(N*(M-1))[:,None]//(M-1)
    vertical_edges[2*np.arange(N*(M-1))+1, 0] = np.arange(N*(M-1))%(M-1)
    vertical_edges[2*np.arange(N*(M-1))+1, 1] = np.arange(N*(M-1))%(M-1) + 1
    
    horizontal_edges = np.zeros((2*M*(N-1), 2))
    horizontal_edges[2*np.arange(M*(N-1)), 0] = np.arange(M*(N-1))%(N-1)
    horizontal_edges[2*np.arange(M*(N-1)), 1] = np.arange(M*(N-1))%(N-1) + 1
    horizontal_edges[2*np.arange(M*(N-1))+1] = np.arange(M*(N-1))[:,None]//(N-1)
    
    # flipping to match imshow axis
    if not plot_potential:
        vertical_edges[2*np.arange(N*(M-1))+1] = M-1 - vertical_edges[2*np.arange(N*(M-1))+1]
        horizontal_edges[2*np.arange(M*(N-1))+1] = M-1 - horizontal_edges[2*np.arange(M*(N-1))+1]

    # edges translation to match with imshow
    vertical_edges -= 0.5
    horizontal_edges -= 0.5

    horizontal_edge_mask = np.repeat(abs(c[:M*(N-1)]) > threshold, 2)
    vertical_edge_mask = np.repeat(abs(c[M*(N-1):]) > threshold, 2)
    
    horizontal_plot = horizontal_edges[horizontal_edge_mask]
    vertical_plot = vertical_edges[vertical_edge_mask]
    
    horizontal_abs = np.abs(c[:M*(N-1)])
    vertical_abs = np.abs(c[M*(N-1):])
    max_abs = max(np.max(horizontal_abs), np.max(vertical_abs))

    horizontal_alpha = horizontal_abs[horizontal_abs > threshold]/max_abs
    horizontal_color = list(zip(["r"]*np.sum(horizontal_abs > threshold), horizontal_alpha))
    vertical_alpha = vertical_abs[vertical_abs > threshold]/max_abs
    vertical_color = list(zip(["r"]*np.sum(vertical_abs > threshold), vertical_alpha))
    
    for i in range(int(len(horizontal_plot)/2)):
        plt.plot(*horizontal_plot[2*i:2*(i+1)], color=horizontal_color[i])
    for i in range(int(len(vertical_plot)/2)):
        plt.plot(*vertical_plot[2*i:2*(i+1)], color=vertical_color[i])

    plt.show()
    plt.close()

def build_filter(N, M):
    # defining filter for inverse Neumann Laplacian
    x = np.reshape(np.arange(N-1),(1,N-1))
    y = np.reshape(np.arange(M-1),(M-1,1))
    x = np.ones((M-1,1))*x
    y = y*np.ones((1,N-1))
    Filt = 2*((np.cos(np.pi*x/(N-1))-1) + (np.cos(np.pi*y/(M-1))-1))
    Filt = 1./(5e-14-Filt)
    Filt[0,0] = 0

    return Filt

def chain_to_vf(N, M, c):
    #chain to vector field
    z = np.zeros((M,N,2))
    z[:,:-1,1] = np.reshape(c[:M*(N-1)], (M,N-1))
    vertical_numbering_edges = M*(N-1) + np.resize(np.arange(M-1)[:,None] + (M-1)*np.arange(N)[None,:], N*(M-1))
    z[:-1,:,0] = np.reshape(c[vertical_numbering_edges], (M-1,N))
    return z

def vf_to_chain(N, M, v):
    #vector field to chain
    m = np.zeros(N*(M-1) + M*(N-1))
    m[:M*(N-1)] = np.reshape(v[:,:-1,1], M*(N-1))
    vertical_numbering_edges = M*(N-1) + np.resize(np.arange(M-1)[:,None] + (M-1)*np.arange(N)[None,:], N*(M-1))
    m[vertical_numbering_edges] = np.reshape(v[:-1,:,0], N*(M-1))
    return m

def simple_curve(N, M, points):
    # building simple curve from sources to sinks
    Filt = build_filter(N, M)
    mu = np.zeros((M-1,N-1))
    for i in range(len(points)//2):
        #number of points should be even (source and sink couples)
        for j in 2*i, 2*i+1:
            if points[j][2] == 0:
                mu[int(points[j][1]), int(points[j][0])] += 1
            else:
                mu[int(points[j][1]), int(points[j][0])] -= 1
    
    u0 = sft.dctn(sft.dctn(mu, type=2)*Filt, type=3)/(4*(M-1)*(N-1))
    m0 = np.stack([np.vstack([u0[1:,:] - u0[:-1,:], np.zeros((1,N-1))]), np.hstack([u0[:,1:] - u0[:,:-1], np.zeros((M-1,1))])], axis=-1)
    return m0

def proj_div(N, M, m):
    #projects onto free divergence fields
    Filt = build_filter(N, M)
    
    u = np.vstack([m[0,:,0], m[1:-1,:,0]-m[0:-2,:,0], -m[-2,:,0]]) + np.hstack([np.reshape(m[:,0,1],(M-1,1)), m[:,1:-1,1]-m[:,0:-2,1], np.reshape(-m[:,-2,1], (M-1,1))])
    u = -sft.dctn(sft.dctn(u,type=2)*Filt,type=3)/(4*(M-1)*(N-1)) # -(D^*D)^{-1} [-D^* z]

    dux = np.vstack([u[1:,:]-u[:-1,:], np.zeros((1,N-1))])
    duy = np.hstack([u[:,1:]-u[:,:-1], np.zeros((M-1,1))])
    m[:,:,0] = m[:,:,0]-dux
    m[:,:,1] = m[:,:,1]-duy
    return m

def integrate_curves(N, M, m, points, threshold_1, threshold_2):
    #obtain the list of paired Dirac masses from the vector fields
    dirac_pairs = []
    integrated_curve = []
    for i in np.where(np.array(points)[:,-1] == 1)[0]:
        #looking at sources, and then finding the corresponding sinks
        source = list(reversed(points[i][:-1]))
        curve = [(source,1)]
        integrated_curve_partial = []
        n = 0
        while n < len(curve):
            current_point, current_coefficient = curve[n]
            escape_sum = 0
            neighbor_edges = []
            directions = []
            if current_point[0] > 0:
                neighbor_edges.append((current_point[0]-1, current_point[1],0))
                directions.append([-1,0])
            if current_point[1] > 0:
                neighbor_edges.append((current_point[0], current_point[1]-1,1))
                directions.append([0,-1])
            if current_point[0] < N-2:
                neighbor_edges.append((current_point[0], current_point[1],0))
                directions.append([1,0])
            if current_point[1] < M-2:
                neighbor_edges.append((current_point[0], current_point[1],1))
                directions.append([0,1])
            for k in range(len(directions)):
                neighbor = np.sum(directions[k])*m[neighbor_edges[k]]
                if neighbor > 0 and escape_sum < current_coefficient:
                    if min(neighbor, current_coefficient - escape_sum) >= threshold_1: #we do not care about very low divergence
                        curve.append(([current_point[0] + directions[k][0], current_point[1] + directions[k][1]] , min(neighbor, current_coefficient - escape_sum)))
                        if n > 0:
                            integrated_curve_partial.append(integrated_curve_partial[n-1]+[np.array(neighbor_edges[k])*np.sum(directions[k])])
                        else:
                            integrated_curve_partial.append([np.array(neighbor_edges[k])*np.sum(directions[k])])

                    m[neighbor_edges[k]] -= np.sum(directions[k])*min(neighbor, current_coefficient-escape_sum)
                    
                    escape_sum += min(neighbor, current_coefficient-escape_sum)
            if escape_sum < current_coefficient - threshold_2: #threshold_2 to take account of the float error
                if not(current_point == source and current_coefficient - escape_sum > 0.999):
                    dirac_pairs.append((source, current_point, current_coefficient - escape_sum))
                    integrated_curve.append((integrated_curve_partial[n-1] if current_point != source else [], current_coefficient - escape_sum))
                
            n += 1

    dirac_pairs_mult, integrated_curve_mult = [], []
    for i in range(len(dirac_pairs)):
        encountered = False
        for j in range(len(dirac_pairs_mult)):
            if dirac_pairs[i][:-1] == dirac_pairs_mult[j][:-1]:
                dirac_pairs_mult[j] = (dirac_pairs_mult[j][0], dirac_pairs_mult[j][1], dirac_pairs_mult[j][-1] + dirac_pairs[i][-1])
                coords = np.abs(integrated_curve[i][0])
                if len(coords) > 0:
                    integrated_curve_mult[j][coords[:,0], coords[:,1], coords[:,2]] += np.sign(integrated_curve[i][0])[:,0]*dirac_pairs[i][-1]
                encountered = True

        if not encountered:
            curve = np.zeros((M-1,N-1,2))
            if integrated_curve[i][0] != []:
                coords = np.abs(integrated_curve[i][0])
                if len(coords) > 0:
                    curve[coords[:,0], coords[:,1], coords[:,2]] = np.sign(integrated_curve[i][0])[:,0]*dirac_pairs[i][-1]
            dirac_pairs_mult.append(dirac_pairs[i])
            integrated_curve_mult.append(curve)

    return dirac_pairs_mult, integrated_curve_mult

def postprocess_curves(N, M, dirac_pairs, integrated_curve):
    new_points = []
    already_treated = [False]*len(dirac_pairs)
    for i in range(len(dirac_pairs)):
        if not already_treated[i]:
            current_curve = np.abs(integrated_curve[i])
            current_width = np.argwhere(current_curve > 1e-12)[:,:-1] #careful for float error
            current_diracs = list(dirac_pairs[i][:-1])
            j = i+1
            while j < len(dirac_pairs):
                c_j = np.argwhere(np.abs(integrated_curve[j]) > 1e-12)[:,:-1]

                if np.sum(np.prod(np.abs(dirac_pairs[j][0] - current_width) <= 3, axis=1)) and np.sum(np.prod(np.abs(current_diracs[1] - c_j) <= 5, axis=1)) and (not already_treated[j]):
                    current_curve += np.abs(integrated_curve[j])
                    current_width = np.argwhere(current_curve > 1e-12)[:,:-1]
                    current_diracs[1] = dirac_pairs[j][1]
                    already_treated[j] = True
                    j = i

                elif np.sum(np.prod(np.abs(dirac_pairs[j][1] - current_width) <= 3, axis=1)) and np.sum(np.prod(np.abs(current_diracs[0] - c_j) <= 5, axis=1)) and (not already_treated[j]):
                    current_curve += np.abs(integrated_curve[j])
                    current_width = np.argwhere(current_curve > 1e-12)[:,:-1]
                    current_diracs[0] = dirac_pairs[j][0]
                    already_treated[j] = True
                    j = i

                elif np.sum(np.prod(np.abs(current_diracs[0] - c_j) <= 3, axis=1)) and np.sum(np.prod(np.abs(current_diracs[1] - c_j) <= 5, axis=1)) and (not already_treated[j]):
                    current_curve += np.abs(integrated_curve[j])
                    current_width = np.argwhere(current_curve > 1e-12)[:,:-1]
                    current_diracs = list(dirac_pairs[j][:-1])
                    already_treated[j] = True
                    j = i

                elif np.sum(np.prod(np.abs(dirac_pairs[j][0] - current_width) <= 3, axis=1)) and np.sum(np.prod(np.abs(dirac_pairs[j][1] - current_width) <= 5, axis=1)) and (not already_treated[j]):
                    already_treated[j] = True
                    restarted = True
                    j = i

                j += 1
            new_points.append(list(reversed(current_diracs[0]))+[0])
            new_points.append(list(reversed(current_diracs[1]))+[1])
    return new_points
