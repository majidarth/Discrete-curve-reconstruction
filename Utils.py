import numpy as np
import scipy.fftpack as sft
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

def click_points(event, points, points_vf):
    # first click for source, second click for sink
    if event.inaxes:
        points.append([round(event.xdata + 0.5), round(event.ydata + 0.5), len(points)%2 == 0]) # transposing by +0.5 because of imshow
        points_vf.append([int(event.xdata + 0.5), int(event.ydata + 0.5), len(points)%2 == 0])
    else:
        print("Point clicked was not inside of axes")

    print("Number of points clicked:", len(points))

def plot_grid(N, M, click=False, points=None, points_vf=None, color=True):
    x, y = np.meshgrid(np.arange(-0.5, N - 0.5), np.arange(-0.5, M - 0.5)) # transposing by -0.5 to match imshow
    plt.plot(x, y, color="b" if color else "w")
    plt.plot(np.transpose(x), np.transpose(y), color="b" if color else "w")
    
    if click:
        plt.connect("button_press_event", lambda event: click_points(event, points, points_vf))

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

def simple_curve(N, M, points, chain=True, smooth=False):
    # building simple curves from source to sink
    if chain:
        # changing N and M to convert chain to vf
        N += 1
        M += 1
    
    Filt = build_filter(N, M)
    mu = np.zeros((M-1,N-1))
    for i in range(len(points)//2):
        # only taking into account an even number of points
        mu[points[2*i][1], points[2*i][0]] += 1
        mu[points[2*i+1][1], points[2*i+1][0]] += -1
        if smooth:
            # smoothening out source and sink
            if points[2*i][1] < M-2:
                mu[points[2*i][1]+1, points[2*i][0]] += 0.5
                if points[2*i][0] < N-2:
                    mu[points[2*i][1]+1, points[2*i][0]+1] += 0.25
                if points[2*i][0] > 0:
                    mu[points[2*i][1]+1, points[2*i][0]-1] += 0.25
            if points[2*i][1] > 0:
                mu[points[2*i][1]-1, points[2*i][0]] += 0.5
                if points[2*i][0] < N-2:
                    mu[points[2*i][1]-1, points[2*i][0]+1] += 0.25
                if points[2*i][0] > 0:
                    mu[points[2*i][1]-1, points[2*i][0]-1] += 0.25
            if points[2*i][0] < N-2:
                mu[points[2*i][1], points[2*i][0]+1] += 0.5
            if points[2*i][0] > 0:
                mu[points[2*i][1], points[2*i][0]-1] += 0.5

            if points[2*i+1][1] < M-2:
                mu[points[2*i+1][1]+1, points[2*i+1][0]] += -0.5
                if points[2*i+1][0] < N-2:
                    mu[points[2*i+1][1]+1, points[2*i+1][0]+1] += -0.25
                if points[2*i+1][0] > 0:
                    mu[points[2*i+1][1]+1, points[2*i+1][0]-1] += -0.25
            if points[2*i+1][1] > 0:
                mu[points[2*i+1][1]-1, points[2*i+1][0]] += -0.5
                if points[2*i+1][0] < N-2:
                    mu[points[2*i+1][1]-1, points[2*i+1][0]+1] += -0.25
                if points[2*i+1][0] > 0:
                    mu[points[2*i+1][1]-1, points[2*i+1][0]-1] += -0.25
            if points[2*i+1][0] < N-2:
                mu[points[2*i+1][1], points[2*i+1][0]+1] += -0.5
            if points[2*i+1][0] > 0:
                mu[points[2*i+1][1], points[2*i+1][0]-1] += -0.5

    
    u0 = sft.dctn(sft.dctn(mu, type=2)*Filt, type=3)/(4*(M-1)*(N-1))
    m0 = np.stack([np.vstack([u0[1:,:] - u0[:-1,:], np.zeros((1,N-1))]), np.hstack([u0[:,1:] - u0[:,:-1], np.zeros((M-1,1))])], axis=-1)

    if chain:
        return vf_to_chain(N-1, M-1, m0)
    else:
        return m0

def proj_div(N, M, m, chain=True):
    # this projects onto free divergence fields
    if chain:
        Filt = build_filter(N+1, M+1)
        z = chain_to_vf(N, M, m)
    else:
        Filt = build_filter(N, M)
        z = m
        # changing N and M to take into account bigger grid in chain case
        N = N-1
        M = M-1
    
    u = np.vstack([z[0,:,0], z[1:-1,:,0]-z[0:-2,:,0], -z[-2,:,0]]) + np.hstack([np.reshape(z[:,0,1],(M,1)), z[:,1:-1,1]-z[:,0:-2,1], np.reshape(-z[:,-2,1], (M,1))])
    u = -sft.dctn(sft.dctn(u,type=2)*Filt,type=3)/(4*M*N) # -(D^*D)^{-1} [-D^* z]

    dux = np.vstack([u[1:,:]-u[:-1,:], np.zeros((1,N))])
    duy = np.hstack([u[:,1:]-u[:,:-1], np.zeros((M,1))])
    z[:,:,0] = z[:,:,0]-dux
    z[:,:,1] = z[:,:,1]-duy
    
    if chain:
        m = vf_to_chain(N, M, z)
        return m
    else:
        return z
