import numpy as np
import scipy.fftpack as sft
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

def click_points(event, points, chain):
    # first click for source, second click for sink
    if chain:
        fit = round
    else:
        fit = int
    if len(points) == 0:
        points.append([fit(event.xdata + 0.5), fit(event.ydata + 0.5), +1]) # transposing by +0.5 because of imshow
    elif len(points) == 1:
        points.append([fit(event.xdata + 0.5), fit(event.ydata + 0.5), -1]) # transposing by +0.5 because of imshow

def plot_grid(N, M, click=False, points=None, color=True, chain=True):
    x, y = np.meshgrid(np.arange(-0.5, N - 0.5), np.arange(-0.5, M - 0.5)) # transposing by -0.5 to match imshow
    plt.plot(x, y, color="b" if color else "w")
    plt.plot(np.transpose(x), np.transpose(y), color="b" if color else "w")
    
    if click:
        plt.connect("button_press_event", lambda event: click_points(event, points, chain))

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
    
def simple_curve(N, M, points, chain=True):
    # building a simple curve from source to sink
    if chain:
        m0 = np.zeros(M*(N-1) + (N-1)*M)
        
        if points[1][0] > points[0][0]:
            m0[points[0][1]*(N-1) + points[0][0]:points[0][1]*(N-1) + points[1][0]] = 1
        else:
            m0[points[0][1]*(N-1) + points[1][0]:points[0][1]*(N-1) + points[0][0]] = -1

        if points[1][1] > points[0][1]:
            m0[M*(N-1) + (M-1)*max(points[0][0], points[1][0]) + points[0][1]:M*(N-1) + (M-1)*max(points[0][0], points[1][0]) + points[1][1]] = 1
        else:
            m0[M*(N-1) + (M-1)*max(points[0][0], points[1][0]) + points[1][1]:M*(N-1) + (M-1)*max(points[0][0], points[1][0]) + points[0][1]] = -1

    else:
        Filt = build_filter(N, M)
        mu = np.zeros((M-1,N-1))
        mu[points[0][1], points[0][0]] = 1
        mu[points[1][1], points[1][0]] = -1
        u0 = sft.dctn(sft.dctn(mu, type=2)*Filt, type=3)/(4*(M-1)*(N-1))
        m0 = np.stack([np.vstack([u0[1:,:] - u0[:-1,:], np.zeros((1,N-1))]), np.hstack([u0[:,1:] - u0[:,:-1], np.zeros((M-1,1))])], axis=-1)
        
    return m0

def chain_to_vf(N, M, c):
    z = np.zeros((M,N,2))
    z[:,:-1,1] = np.reshape(c[:M*(N-1)], (M,N-1))
    vertical_numbering_edges = M*(N-1) + np.resize(np.arange(M-1)[:,None] + (M-1)*np.arange(N)[None,:], N*(M-1))
    z[:-1,:,0] = np.reshape(c[vertical_numbering_edges], (M-1,N))
    return z

def vf_to_chain(N, M, v):
    m = np.zeros(N*(M-1) + M*(N-1))
    m[:M*(N-1)] = np.reshape(v[:,:-1,1], M*(N-1))
    vertical_numbering_edges = M*(N-1) + np.resize(np.arange(M-1)[:,None] + (M-1)*np.arange(N)[None,:], N*(M-1))
    m[vertical_numbering_edges] = np.reshape(v[:-1,:,0], N*(M-1))
    return m

def proj_div(N, M, m, chain=True):
    # this projects onto free divergence fields
    if chain:
        Filt = build_filter(N+1, M+1)
        z = chain_to_vf(N, M, m)
    else:
        Filt = build_filter(N, M)
        z = m
        # changing M and N to take into account bigger grid in chain case
        M = M-1
        N = N-1
    
    u = np.vstack([z[0,:,0], z[1:-1,:,0]-z[0:-2,:,0], -z[-2,:,0]]) + np.hstack([np.reshape(z[:,0,1],(M,1)), z[:,1:-1,1]-z[:,0:-2,1], np.reshape(-z[:,-2,1], (M,1))])
    u = -sft.dctn(sft.dctn(u,type=2)*Filt,type=3)/(4*M*N) # -(D^*D)^{-1} [-D^* z]

    dux = np.vstack([ u[1:,:]-u[:-1,:], np.zeros((1,N))])
    duy = np.hstack([ u[:,1:]-u[:,:-1], np.zeros((M,1))])
    z[:,:,0] = z[:,:,0]-dux
    z[:,:,1] = z[:,:,1]-duy
    
    if chain:
        m = vf_to_chain(N, M, z)
        return m
    else:
        return z
