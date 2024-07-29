import numpy as np
import scipy.fftpack as sft
import matplotlib.pyplot as plt

## this projects onto Free Divergence fields
def ProjDiv(z): # z = array[M,N,2]
    # the last line/column of zx, zy have to be zero
    [M,N,D] = z.shape
    # z[-1,:,0]=0
    # z[:,-1,1]=0

    # the filter could be built separately beforehand
    x = np.reshape(np.arange(N),(1,N))
    y = np.reshape(np.arange(M),(M,1))
    x = np.ones((M,1))*x;
    y = y*np.ones((1,N));
    Filt = 2*((np.cos(np.pi*x/N)-1) + (np.cos(np.pi*y/M)-1))
    Filt = 1./(5e-14-Filt)
    Filt[0,0]=0
    # end of filter
    
    u = np.vstack([z[0,:,0], z[1:-1,:,0]-z[0:-2,:,0], -z[-2,:,0]]) + np.hstack([np.reshape(z[:,0,1],(M,1)), z[:,1:-1,1]-z[:,0:-2,1], np.reshape(-z[:,-2,1], (M,1))])
    u = -sft.dctn(sft.dctn(u,type=2)*Filt,type=3)/(4*M*N) # -(D^*D)^{-1} [-D^* z]

    dux = np.vstack([ u[1:,:]-u[:-1,:], np.zeros((1,N))])
    duy = np.hstack([ u[:,1:]-u[:,:-1], np.zeros((M,1))])
    z[:,:,0] = z[:,:,0]-dux
    z[:,:,1] = z[:,:,1]-duy
    return z

g = plt.imread("virgule.png")
sh = g.shape
if len(sh) == 3:
    g = g[:,:,0:3].mean(axis=2)
M = sh[0]
N = sh[1]

# the filter could be built separately beforehand
x = np.reshape(np.arange(N),(1,N))
y = np.reshape(np.arange(M),(M,1))
x = np.ones((M,1))*x
y = y*np.ones((1,N))
Filt = 2*((np.cos(np.pi*x/N)-1) + (np.cos(np.pi*y/M)-1))
Filt = 1./(5e-14-Filt)
Filt[0,0]=0
# end of filter

weights = 1/(.1+g) # +x*x+y*y does not work # find some image
mu = np.zeros((M,N))
mu[M//20,5*N//10]=1
mu[4*M//10,19*N//20]=-1

u0 = sft.dctn(sft.dctn(mu,type=2)*Filt,type=3)/(4*M*N)
z0 = np.stack([ np.vstack([ u0[1:,:]-u0[:-1,:], np.zeros((1,N))]), np.hstack([ u0[:,1:]-u0[:,:-1], np.zeros((M,1))]) ], axis=-1)

z = z0.copy()
p = np.zeros((M,N,2))
oldz = z.copy()

sigma=np.sqrt(np.sqrt(M*N))
tau=.99/sigma

for i in range(400):
    p = p + sigma*(2*z-oldz)
    p = p/(np.maximum(1,np.hypot(p[:,:,0],p[:,:,1])/weights)[:,:,None])

    oldz = z.copy()
    z = z - tau*p
    z = z0 + ProjDiv(z-z0)
    
normz = np.hypot(z[:,:,0],z[:,:,1])
ax = plt.figure().add_subplot(projection='3d')
surf = ax.plot_surface(x,y,normz)
plt.show()

plt.figure()
plt.imshow(normz)
plt.figure()
plt.imshow(weights)
plt.show()
