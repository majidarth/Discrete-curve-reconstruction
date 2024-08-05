import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    # parsing arguments for image size, source and implementation
    parser = argparse.ArgumentParser(prog="Discrete curve reconstruction", description="Either pass an image as argument or define a potential by clicking on the tiles of a grid")
    parser.add_argument("-N", "--width", required=True, help="width of the grid")
    parser.add_argument("-M", "--height", required=True, help="height of the grid")
    parser.add_argument("--n_iter", required=True, help="number of iterations for Chambolle-Pock")
    parser.add_argument("-i", "--image", required=False, help="input image path, image must be larger than width-1 and height-1, and in RGBA format")
    parser.add_argument("--implementation", required=True, help="algorithm implementation (A, B or C)")
    args = vars(parser.parse_args())
    
    return args

def potential_init(N, M, image):
    # rectangular "N*M nodes" grid
    # defining the grid potential
    grid_potential = np.ones((M-1, N-1))

    if image != None:  # assembling potential from image
        img = (1 - np.mean(np.array(plt.imread(image))[:,:,1:3], axis=2))**4 # arbitraty rescaling of the greyscale image
        pixel_size_w = round(img.shape[1]/(N-1))
        pixel_size_h = round(img.shape[0]/(M-1))
        
        for i in range(M-1):
            for j in range(N-1):
                grid_potential[i,j] = np.mean(img[pixel_size_h*i:pixel_size_h*i + pixel_size_h, pixel_size_w*j:pixel_size_w*j + pixel_size_w])

    else:
    # clicking a grid to define the potential
    # default level is 1, click once to set to 0.5, twice to set to 0
        def click_potential(event):
            grid_potential[int(M-1 - event.ydata), int(event.xdata)] = max(grid_potential[int(M-1 - event.ydata), int(event.xdata)] - 0.5, 0) # flipping along y axis to match imshow

        x, y = np.meshgrid(np.arange(N), np.arange(M))
        plt.plot(x, y, color="b")
        plt.plot(np.transpose(x), np.transpose(y), color="b")

        plt.connect("button_press_event", click_potential)
        plt.show()
        plt.close()

    return grid_potential

def edge_potential(N, M, grid_potential):
    g = np.zeros(N*(M-1) + M*(N-1))
    padded = np.pad(grid_potential, 1, mode="edge")
    # horizontal edges
    g[:M*(N-1)] = np.reshape(0.5*padded[:-1,1:-1] + 0.5*padded[1:,1:-1], M*(N-1))
    # vertical edges
    vertical_numbering = M*(N-1) + np.resize(np.arange(M-1)[:,None] + (M-1)*np.arange(N)[None,:], N*(M-1))
    g[vertical_numbering] = np.reshape(0.5*padded[1:-1,:-1] + 0.5*padded[1:-1,1:], N*(M-1))

    return g
