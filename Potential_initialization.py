import argparse
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    # parsing arguments for image size, source and implementation
    parser = argparse.ArgumentParser(prog="Discrete curve reconstruction", description="Either pass an image as argument or define a potential by clicking on the tiles of a grid")
    parser.add_argument("-N", "--width", required=True, help="width of the grid")
    parser.add_argument("-M", "--height", required=True, help="height of the grid")
    parser.add_argument("--n_iter", required=True, help="number of iterations for Chambolle-Pock")
    parser.add_argument("--n_curves", required=False, help="number of curves to find in the image")
    parser.add_argument("-i", "--image", required=False, help="input image path, image must be larger than width-1 and height-1, and in RGBA format")
    parser.add_argument("--implementation", required=True, help="algorithm implementation (A, B, C or D)")
    parser.add_argument("--save_result", required=False, help="save result image", action="store_true")
    parser.add_argument("--save_diracs", required=False, help="save source and sink", action="store_true")
    parser.add_argument("--diracs", required=False, help="numpy matrix file containing source and sink nodes", nargs=2)
    args = vars(parser.parse_args())
    
    return args

def potential_init(N, M, image):
    # rectangular "N*M nodes" grid
    # defining the grid potential
    grid_potential = np.ones((M-1, N-1))

    if image != None:  # assembling potential from image
        img = (1 - np.mean(np.array(plt.imread(image)), axis=2))**4 # arbitrary rescaling of the greyscale image
        # img = (np.mean(np.array(plt.imread(image)), axis=2))**4 # arbitrary rescaling of the greyscale image
        
        pixel_size_w = int(img.shape[1]/(N-1))
        pixel_size_h = int(img.shape[0]/(M-1))
        
        for i in range(M-1):
            for j in range(N-1):
                grid_potential[i,j] = np.mean(img[pixel_size_h*i:pixel_size_h*i + pixel_size_h, pixel_size_w*j:pixel_size_w*j + pixel_size_w])

    else:
    # clicking a grid to define the potential
    # default level is 1, click once to set to 0.5, twice to set to 0
        def click_potential(event):
            if event.inaxes:
                grid_potential[int(M-1 - event.ydata), int(event.xdata)] = max(grid_potential[int(M-1 - event.ydata), int(event.xdata)] - 0.5, 0) # flipping along y axis to match imshow
            else:
                print("Point clicked was not inside of axes")

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
