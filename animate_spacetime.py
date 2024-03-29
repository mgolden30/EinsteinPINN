import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the trained neural network
tetradnet = torch.load("network_output/tetradnet_collision.pth").to('cpu')
tetradnet.eval()  # Set the network to evaluation mode

# Define parameters for the 2D grid
r_min = 0.5  # Don't resolve closer than this to the singularity
r_max = 10
t_max = 5

# Give multiple black hole positions
pos = np.zeros((2, 3))
pos[0, :] = [1.0, 0.0, 0.0]
pos[1, :] = [-1.0, 0.0, 0.0]

nx = 128
nt = 32

# Generate the 2D grid
grid_1d = torch.linspace(-r_max, r_max, nx)
[xx,yy] = torch.meshgrid( grid_1d, grid_1d )
xx = torch.reshape( xx, [-1,1] )
yy = torch.reshape( yy, [-1,1] )
zz = 0*xx

eta = torch.tensor( [-1,1,1,1] ) #just store it as a vector

for t in torch.linspace(0, t_max, nt):
    #Make an input matrix x for current time
    tt = 0*xx + t
    x  = torch.cat( (tt,xx,yy,zz), dim=1 )

    # Evaluate the network on the grid
    with torch.no_grad():
        e = tetradnet(x)
    g = torch.einsum( 'bmi,bni,i->bmn', e, e, eta )

    # Convert the output to a numpy array
    g = g.cpu().numpy()

    # Plot the output image
    print(t)
    mask = (xx*xx + yy*yy > r_max*r_max)
    mask = np.squeeze(mask)
    g[mask,:,:] = 0
    
    this_frame = np.reshape( g[:,0,0], [nx, nx] )
    plt.imshow( this_frame, cmap='viridis')  # You can choose a different colormap if needed
    plt.colorbar()  # Add a colorbar for reference
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Neural Network Output')
    plt.show()
