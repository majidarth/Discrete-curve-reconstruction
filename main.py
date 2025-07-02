import matplotlib.pyplot as plt
import numpy as np
import datetime

import Potential_initialization
import Utils

import Implementation

args = Potential_initialization.parse_args()
N = int(args["width"])
M = int(args["height"])
n_iter = int(args["n_iter"])
if args["n_curves"] != None:
    n_curves = int(args["n_curves"])
else:
    n_curves = 1
image = args["image"]

#initializing grid potential
grid_potential = Potential_initialization.potential_init(N, M, image)

#visualizing grid potential
plt.imshow(grid_potential, cmap="gray")
plt.show()
plt.close()

# clicking source and sink points
if args["diracs"] == None and args["max_pot"] == None:
    print("Click on source and sink points on the blue grid")
    grid_potential_rgb = np.stack([grid_potential, grid_potential, grid_potential], axis=-1)
    points = []
    for j in range(n_curves):
        plt.imshow(grid_potential_rgb)
        Utils.plot_grid(N, M, click=True, points=points)
        plt.show()
        plt.close()
        
elif args["diracs"] != None:
    points = np.load(args["diracs"])

if args["max_pot"] != None:
    max_pot = float(args["max_pot"]) #points are chosen randomly where potential is under max_pot and not on the border
    n_points = 2*n_curves #number of points (source and sink)
    close = args["points_close"] #whether or not source and sink couples need to be chosen close to one another
    
    potential_under_max = np.argwhere(grid_potential < max_pot)
    potential_under_max = potential_under_max[(potential_under_max[:,0] > 0)*(potential_under_max[:,0] < M-2-2*close)*(potential_under_max[:,1] > 0)*(potential_under_max[:,1] < N-2-2*close)]
    args_source = np.random.choice(len(potential_under_max), size=n_points)
    points_source = np.flip(potential_under_max[args_source],axis=1).tolist() #need to flip x,y
    if close:
        points = [points_source[i//2]+[0] if i%2==0 else [points_source[i//2][0], points_source[i//2][1]+1, 1] for i in range(n_points)]
    else:
        points = [points_source[i]+[0] if i%2==0 else points_source[i]+[1] for i in range(n_points)]

if args["save_diracs"]:
    np.save(str(datetime.datetime.now())+"_diracs.npy", points)

if args["n_moves"] == None or int(args["n_moves"]) == 0:
    Implementation.curve_reconstruction(N, M, points, grid_potential, n_iter, save=args["save_result"])

else:
    Implementation.curve_discovery(N, M, points, grid_potential, n_iter, int(args["n_moves"]), float(args["max_pot"]))
