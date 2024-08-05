import matplotlib.pyplot as plt

import Potential_initialization
import Utils

import Implementation_A
import Implementation_B
import Implementation_C

args = Potential_initialization.parse_args()
N = int(args["width"])
M = int(args["height"])
n_iter = int(args["n_iter"])
image = args["image"]
implementation = globals()[args["implementation"]]

# initializing grid potential
grid_potential = Potential_initialization.potential_init(N, M, image)

# clicking source and sink points
points = []
plt.imshow(grid_potential, cmap="gray")
Utils.plot_grid(N, M, click=True, points=points)
plt.show()
plt.close()

# curve reconstruction with chosen implementation
implementation.curve_reconstruction(N, M, points, grid_potential, n_iter)
