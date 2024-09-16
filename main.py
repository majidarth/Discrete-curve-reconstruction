import matplotlib.pyplot as plt
import numpy as np

import Potential_initialization
import Utils

import Implementation_A
import Implementation_B
import Implementation_C
import Implementation_D

args = Potential_initialization.parse_args()
N = int(args["width"])
M = int(args["height"])
n_iter = int(args["n_iter"])
if args["n_curves"] != None:
    n_curves = int(args["n_curves"])
else:
    n_curves = 1
image = args["image"]
if args["implementation"] != "all":
    implementation = globals()["Implementation_" + args["implementation"]]

# initializing grid potential
grid_potential = Potential_initialization.potential_init(N, M, image)

# visualizing grid potential
plt.imshow(grid_potential, cmap="gray")
plt.show()
plt.close()

# clicking source and sink points
grid_potential_rgb = np.stack([grid_potential, grid_potential, grid_potential], axis=-1)
points = []
points_vf = []
for j in range(n_curves):
    plt.imshow(grid_potential_rgb)
    Utils.plot_grid(N, M, click=True, points=points, points_vf=points_vf)
    plt.show()
    plt.close()
    for i in range(len(points)):
        small_grid_x = np.arange(-1,2)[((points[i][1] + np.arange(-1,2)) >= 0)*((points[i][1] + np.arange(-1,2)) <= M-1)]
        small_grid_y = np.arange(-1,2)[((points[i][0] + np.arange(-1,2)) >= 0)*((points[i][1] + np.arange(-1,2)) <= N-1)]
        grid_potential_rgb[points[i][1] + small_grid_x[:,None], points[i][0] + small_grid_y[None,:]] = [1, 0, 0]

# curve reconstruction with chosen implementation
if args["implementation"] == "all":
    Implementation_A.curve_reconstruction(N, M, points, grid_potential, n_iter, save=args["save"])
    Implementation_B.curve_reconstruction(N, M, points_vf, grid_potential, n_iter, save=args["save"])
    Implementation_C.curve_reconstruction(N, M, points, grid_potential, n_iter, save=args["save"])
    Implementation_C.curve_reconstruction(N, M, points, grid_potential, n_iter, smooth=True, save=args["save"])
    Implementation_D.curve_reconstruction(N, M, points_vf, grid_potential, n_iter, save=args["save"])
else:
    implementation.curve_reconstruction(N, M, points_vf if args["implementation"] in ("B", "D") else points, grid_potential, n_iter, save=args["save"])
