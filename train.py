'''
PURPOSE:
This script trains a randomly initialized network to learn a metric that satisfies the Einstein Field Equations.
'''

import torch

import torch.optim as optim
from torch import nn
from lib.model import TetradNetwork_V1, EinsteinPINN

import lib.utils as utils


device = utils.check_for_GPU()

#PUT ALL PARAMETERS HERE
epochs = 64 #number of training steps
learning_rate = 1e-2
num_training  = 1024 #number of training points

#Seed for reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#Pick training data
x_train = utils.sample_uniform_cube( num_training ) #sample from [-1,1]^4
x_test  = utils.sample_uniform_cube( num_training ) #sample from [-1,1]^4

#Create a new network for computing a random tetrad
tetradnet = TetradNetwork_V1().to(device)
pinn      = EinsteinPINN(tetradnet).to(device) #create a "pinn" with the tetradnet as a subnetwork

#define loss scaling and the optimizer of interest
criterion = nn.L1Loss()
optimizer = optim.Adam( pinn.parameters(), lr=learning_rate )
    
loss_history = torch.zeros( (epochs) )

for epoch in range(epochs):
    # Forward pass to compute spacetime curvature
    ricci, riemann, _, _, _ = pinn.forward(x_train)

    #We want the Ricci tensor to vanish. I am weighting by the Riemann tensor
    #Both are projected into Minkowski space with the tetrad, so this loss is coordinate invariant
    #It is NOT gauge invariant though. Local Lorentz transformations of the tetrad will change it.
    err  = ricci/torch.mean( torch.abs(riemann), dim=(0,1,2,3,4) )

    #turn the output into a loss
    loss = criterion(err, torch.zeros_like(err))
    loss_history[epoch] = loss.detach()
    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    #LEARN
    optimizer.zero_grad()
    loss.backward(retain_graph=True)   
    optimizer.step()

    #Reduce learning rate every so often if you want
    if( (epoch % 64 == 0) & (epoch > 0) ):
        learning_rate = learning_rate/2
        optimizer = optim.Adam( pinn.parameters(), lr=learning_rate )

#save everything out
output_filename = "./network_output/tetradnet" #utils.save_network will add file extensions for the appropriate output files.
utils.save_network( tetradnet, x_train, x_test, loss_history, output_filename )