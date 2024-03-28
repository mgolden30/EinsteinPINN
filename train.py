'''
PURPOSE:
This script trains a randomly initialized network to learn a metric that satisfies the Einstein Field Equations.
'''

import torch

import torch.optim as optim
from torch import nn
from lib.model import TetradNetwork_V1, MultiBlackHoleNetwork
from lib.losses import loss_black_hole
import lib.utils as utils
import numpy as np

device = utils.check_for_GPU()

#PUT ALL PARAMETERS HERE
epochs = 3*64 #number of training steps
learning_rate = 1e-2
num_training  = 1024 #number of training points

#architecture parameters
num_layers = 3
feature_size = 64

#Seed for reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#Pick training data
#x_train = utils.sample_uniform_cube( num_training ) #sample from [-1,1]^4
#x_test  = utils.sample_uniform_cube( num_training ) #sample from [-1,1]^4

r_min = 0.5 #Don't resolve closer than this to the singularity 
r_max = 10
t_max = 5

#give multiple black hole positions
pos = np.zeros( (2,3) )
pos[0,:] = [ 1.0, 0.0, 0.0 ]
pos[1,:] = [-1.0, 0.0, 0.0 ]

x_train = utils.sample_black_holes(num_training, t_max, r_min, r_max, pos)
x_test  = utils.sample_black_holes(num_training, t_max, r_min, r_max, pos)

x0 = x_train.clone().detach()
x0.requires_grad = False
x0[:,0] = 0 #set t=0 manually

#Create a new network for computing a random tetrad
tetradnet = TetradNetwork_V1(num_layers, feature_size).to(device)
schwarz   = MultiBlackHoleNetwork(pos).to(device)

#tetradnet = torch.load("network_output/tetradnet.pth")

#define loss scaling and the optimizer of interest
criterion = nn.L1Loss()
optimizer = optim.Adam( tetradnet.parameters(), lr=learning_rate )
    
loss_history = torch.zeros( (epochs) )

for epoch in range(epochs):
    err = loss_black_hole( tetradnet, x_train, schwarz, x0 )

    #turn the error into a loss
    loss = criterion(err, torch.zeros_like(err))
    loss_history[epoch] = loss.detach()
    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

    #LEARN
    optimizer.zero_grad()
    loss.backward(retain_graph=True)   
    optimizer.step()

    #Reduce learning rate every so often if you want
    if( (epoch % 64 == 0) & (epoch > 0) ):
        learning_rate = learning_rate/10
        optimizer = optim.Adam( tetradnet.parameters(), lr=learning_rate )

#save everything out
output_filename = "./network_output/tetradnet_normal" #utils.save_network will add file extensions for the appropriate output files.
utils.save_network( tetradnet, x_train, x_test, loss_history, output_filename )