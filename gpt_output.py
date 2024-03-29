import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the trained neural network
tetradnet = torch.load("network_output/tetradnet_binary.pth").to('cpu')
tetradnet.eval()  # Set the network to evaluation mode

# Define parameters for the 2D grid
r_min = 0.5 #Don't resolve closer than this to the singularity 
r_max = 15
t_max = 10

#give multiple black hole positions
pos = np.zeros( (2,3) )
pos[0,:] = [ 3.0, 0.0, 0.0 ]
pos[1,:] = [-3.0, 0.0, 0.0 ]

nx = 128
nt = 32

# Generate the 2D grid
grid_1d = torch.linspace(-r_max, r_max, nx)
[xx, yy] = torch.meshgrid(grid_1d, grid_1d, indexing="xy")
xx = torch.reshape(xx, [-1, 1])
yy = torch.reshape(yy, [-1, 1])
zz = 0 * xx

eta = torch.tensor([-1, 1, 1, 1])  # just store it as a vector

# Create a figure and axis for the animation
fig, ax = plt.subplots()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('g_{tt}')

# Create a dummy image for the colorbar
dummy_im = ax.imshow(np.zeros((nx, nx)), cmap='bwr', vmin=-1, vmax=1)
plt.colorbar(dummy_im)  # Add a colorbar using the dummy image

def update(t):
    # Make an input matrix x for current time
    tt = 0 * xx + t
    x = torch.cat((tt, xx, yy, zz), dim=1)

    # Evaluate the network on the grid
    with torch.no_grad():
        e = tetradnet(x)
    g = torch.einsum('bmI,bnI,I->bmn', e, e, eta)

    # Convert the output to a numpy array
    g = g.cpu().numpy()

    # Mask out regions outside r_max
    mask = (xx * xx + yy * yy + zz * zz > r_max * r_max)
    mask = np.squeeze(mask)
    g[mask, :, :] = np.nan

    #Also mask out black hole positions
    for bh_pos in pos:
        r = torch.norm( x[:, 1:] - bh_pos, dim=1)  # Compute distance to each black hole
        mask = r < r_min
        g[mask, :, :] = np.nan

    this_frame = np.reshape(g[:, 1, 1], [nx, nx])
    im = ax.imshow(this_frame, cmap='bwr')
    return im,

# Create the animation
ani = FuncAnimation(fig, update, frames=np.linspace(0, t_max, nt), blit=True)
plt.show()
