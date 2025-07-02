## Discrete curve reconstruction
Basic example to recover a single curve, with endpoints clicked by user, in the 100x100 pixel image `virgule.png`:\
`python main.py -N 100 -M 100 --n_iter 3000 -i virgule.png`

Complete list of possible arguments and their description:
|Flag &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|Description|Default value|
|---|---|---|
|`-N`|Width of the grid delimited by the image| **Required**|
|`-M`|Height of the grid delimited by the image|**Required**|
|`--n_iter`|Number of iterations for the Chambolle-Pock algorithm|**Required**|
|`--n_curves`|Number of curves to find in the image|`1`|
|`-i`|Input image path, image must be larger than (N-1)x(M-1), and in RGBA format. If no image is given, user will have to define one manually by clicking in a grid|`None`|
|`--save_result`|Toggle whether or not result image of the reconstructed curve is saved|`False`|
|`--save_diracs`|Toggle whether or not initial Dirac masses are saved in a `.npy` file|`False`|
|`--diracs`|Path of a `.npy` file of previously saved Dirac masses|`None`|
|`--max_pot`|If this value is specified, initial Dirac masses will be chosen at random, at pixels where the image's greyscale level is below this value|`None`|
|`--points_close`|Toggle whether or not, when initial Dirac masses are chosen at random, source and sink pairs are to be chosen at one pixel distance to each other|`False`|
|`--n_moves`|Number of maximum movement steps for Dirac masses|`1`|
