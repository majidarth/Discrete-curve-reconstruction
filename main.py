import matplotlib.pyplot as plt

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
points = []
points_vf = []
plt.imshow(grid_potential, cmap="gray")
Utils.plot_grid(N, M, click=True, points=points, points_vf=points_vf)
plt.show()
plt.close()

# curve reconstruction with chosen implementation
if args["implementation"] == "all":
    Implementation_A.curve_reconstruction(N, M, points, grid_potential, n_iter)
    Implementation_B.curve_reconstruction(N, M, points_vf, grid_potential, n_iter)
    Implementation_C.curve_reconstruction(N, M, points, grid_potential, n_iter)
    # Implementation_C.curve_reconstruction(N, M, points, grid_potential, n_iter, smooth=True)
    Implementation_D.curve_reconstruction(N, M, points_vf, grid_potential, n_iter)
else:
    implementation.curve_reconstruction(N, M, points_vf if args["implementation"] in ("B", "D") else points, grid_potential, n_iter)
