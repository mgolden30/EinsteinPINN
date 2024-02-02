import numpy as np
import torch

import torch.optim as optim
from torch import nn
from lib.model import TetradNetwork_V1, EinsteinPINN
import lib.utils as utils
from scipy.io import savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_folder = "network_output/"

epochs = 32
learning_rate = 1e-1

#For reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

N = 1024
x_train = utils.sample_uniform_cube(N) 
x_test  = utils.sample_uniform_cube(N) 

loss_history = torch.zeros( (epochs) )

tetradnet = torch.load("network_output/tetradnet.pth")
pinn      = EinsteinPINN(tetradnet).to(device)

criterion = nn.L1Loss()
optimizer = optim.LBFGS( pinn.parameters(), lr=1e-2)

def closure():
    optimizer.zero_grad()  # Clear gradients
    ricci, riemann, _, _, _ = pinn.forward(x_train)
    err_einstein = ricci/torch.mean( torch.abs(riemann), dim=(0,1,2,3,4) )
    
    err = err_einstein
    loss = criterion(err, torch.zeros_like(err))
    loss.backward(retain_graph=True)   # compute gradients
    return loss

for epoch in range(epochs):
    # Forward pass
    ricci, riemann, _, _, _ = pinn.forward(x_train)
    err_einstein = ricci/torch.mean( torch.abs(riemann), dim=(0,1,2,3,4) )
    err = err_einstein

    # Compute the loss
    loss = criterion(err, torch.zeros_like(err))  # assuming you want to minimize pinn.forward(xs) to zero
    loss_history[epoch] = loss.detach()

    # Backward pass and optimization step
    optimizer.zero_grad()  # clear previous gradients
    loss.backward(retain_graph=True)   # compute gradients
    optimizer.step(closure)  # update model parameters

    # Print the loss every few epochs
    if epoch % 1 == 0:
        #save_network_output( hydro_model, "torch_output_newton_%d.mat" % (epoch) )
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

output_filename = "./network_output/tetradnet_finetuned" #utils.save_network will add file extensions for the appropriate output files.
utils.save_network( tetradnet, x_train, x_test, loss_history, output_filename )