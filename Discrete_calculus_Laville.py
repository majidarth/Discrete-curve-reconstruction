import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

parser = argparse.ArgumentParser(prog="'Discrete calculus' implementation", description="Either pass an image as argument or define a potential by clicking on the tiles of a grid")
parser.add_argument("-N", "--width", required=True, help="width of the grid")
parser.add_argument("-M", "--height", required=True, help="height of the grid")
parser.add_argument("--n_iter", required=True, help="number of iterations for Chambolle-Pock")
parser.add_argument("-i", "--image", required=False, help="input image path, image must be larger than width-1 and height-1, and in RGBA format")
args = vars(parser.parse_args())

# rectangular "N*M" grid
N = int(args["width"])
M = int(args["height"])
n_nodes = N*M
n_edges = (N-1)*M + N*(M-1)
n_faces = (N-1)*(M-1)

# defining the grid potential
grid_potential = np.ones((M-1, N-1))

if args["image"] != None:
    # assembling potential "g" from image
    image = (1 - np.mean(np.array(plt.imread(args["image"]))[:,:,1:3], axis=2))**4
    pixel_size = round(image.shape[0]/(M-1))

    for i in range(M-1):
        for j in range(N-1):
            grid_potential[i,j] = np.mean(image[pixel_size*i:pixel_size*i + pixel_size, pixel_size*j:pixel_size*j + pixel_size])

else:
    # clicking a grid to define the potential
    # click once to set to 0.5, twice to set to 0
    def click_potential(event):
        grid_potential[int(M-1 - event.ydata), int(event.xdata)] = max(grid_potential[int(M-1 - event.ydata), int(event.xdata)] - 0.5, 0)

    x, y = np.meshgrid(np.arange(N), np.arange(M))
    plt.plot(x, y, color="b")
    plt.plot(np.transpose(x), np.transpose(y), color="b")

    plt.connect("button_press_event", click_potential)
    plt.show()
    plt.close()

# defining the edge potentials
g = np.ones(n_edges)
# horizontal edges
g[:(N-1)*M] = np.reshape(0.5*np.pad(grid_potential, 1, mode="edge")[1:, 1:-1] + 0.5*np.pad(grid_potential, 1, mode="edge")[:-1, 1:-1], (N-1)*M)
# vertical edges
vertical_numbering = (N-1)*M + np.resize(np.arange(M)[:,None] + (M-1)*np.arange(N)[None,:], N*(M-1))
g[vertical_numbering] = np.reshape(0.5*np.pad(grid_potential, 1, mode="edge")[1:-1, 1:] + 0.5*np.pad(grid_potential, 1, mode="edge")[1:-1, :-1], N*(M-1))

# clicking source and sink points
points = []
def click_points(event):
    global points
    #First click for source, second click for sink
    if len(points) == 0:
        points.append((round(event.xdata + 0.5), round(event.ydata + 0.5), +1))
    elif len(points) == 1:
        points.append((round(event.xdata + 0.5), round(event.ydata + 0.5), -1))

def plot_grid(click=False, color=True):
    x, y = np.meshgrid(np.arange(-0.5, N - 0.5), np.arange(-0.5, M - 0.5))
    plt.plot(x, y, color="b" if color else "w")
    plt.plot(np.transpose(x), np.transpose(y), color="b" if color else "w")
    
    if click:
        plt.connect("button_press_event", click_points)

plt.imshow(grid_potential, cmap="gray")
plot_grid(click=True)
plt.show()
plt.close()

def plot_cochain(c, plot_all_edges=True, plot_potential=False, threshold=0.1):
    if plot_all_edges:
        plot_grid(color=True)
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

# plot_cochain(g, plot_all_edges=True, plot_potential=True)
    
# building a simple path from source to sink
m0 = np.zeros(n_edges)
if points[1][0] > points[0][0]:
    m0[points[0][1]*(N-1) + points[0][0]:points[0][1]*(N-1) + points[1][0]] = 1
else:
    m0[points[0][1]*(N-1) + points[1][0]:points[0][1]*(N-1) + points[0][0]] = -1

if points[1][1] > points[0][1]:
    m0[M*(N-1) + (M-1)*max(points[0][0], points[1][0]) + points[0][1]:M*(N-1) + (M-1)*max(points[0][0], points[1][0]) + points[1][1]] = 1
else:
    m0[M*(N-1) + (M-1)*max(points[0][0], points[1][0]) + points[1][1]:M*(N-1) + (M-1)*max(points[0][0], points[1][0]) + points[0][1]] = -1
    
def chambolle_pock(g, m0, n_iter, sigma=.99/(8*np.sqrt(np.sqrt(M*N))), tau=np.sqrt(np.sqrt(M*N)), theta=1):
    old_m = m0.copy()
    new_m = m0.copy()
    m_barre = m0.copy()
    u = np.zeros(n_nodes)
    n = 0
    while n < n_iter:
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
        new_m = (shrink > tau*g + alpha)*(shrink - tau*g - alpha) + (shrink < -tau*g + alpha)*(shrink + tau*g - alpha)
        
        m_barre = new_m + theta*(new_m - old_m)
        n += 1

    energy = np.sum(g*np.abs(new_m))
    print("energy = ", energy)
        
    return new_m

alpha = 0
m = chambolle_pock(g, m0, int(args["n_iter"]))
plot_cochain(m, plot_potential=True, plot_all_edges=True)
plot_cochain(m, plot_potential=False, plot_all_edges=False)
