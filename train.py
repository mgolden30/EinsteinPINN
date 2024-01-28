import numpy as np
import os
import torch

import torch.optim as optim
from torch import nn
from lib.model import TetradNetwork, SchwarzschildTetradNetwork, EinsteinPINN

from scipy.io import savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_folder = "network_output/"

epochs = 128
learning_rate = 1e-2

#For reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

N = 1024*2
x = (2*torch.rand( (N, 4), requires_grad=True ) - 1)

#for testing Schwarzschild
#x = 2.1 + 0.5*torch.rand( (N, 4), requires_grad=True ) 
#tetradnet = SchwarzschildTetradNetwork()

tetradnet = TetradNetwork()
pinn      = EinsteinPINN(tetradnet)

criterion = nn.L1Loss()
optimizer = optim.Adam( pinn.parameters(), lr=learning_rate )
    
loss_history = torch.zeros( (epochs) )

for epoch in range(epochs):
    # Forward pass (with output of PickDomains!)
    ricci, riemann, _, _, _ = pinn.forward(x)

    #Define a normalized error
    err = ricci/torch.mean( torch.abs(riemann), dim=(0,1,2,3,4) )

    # Compute the MSE loss
    loss = criterion(err, torch.zeros_like(err))  # assuming you want to minimize pinn.forward(xs) to zero
    loss_history[epoch] = loss.detach()

    # clear previous gradients
    optimizer.zero_grad()
    loss.backward(retain_graph=True)   
        
    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
    
    # update model parameters
    optimizer.step()


ricci, riemann, wald_1, wald_2, w = pinn.forward(x)

ricci = ricci.clone().to("cpu").detach()
riemann= riemann.clone().to("cpu").detach()
e     = tetradnet.forward(x).clone().to("cpu").detach()
x     = x.clone().to("cpu").detach()
wald_1= wald_1.clone().to("cpu").detach()
wald_2= wald_2.clone().to("cpu").detach()
w     = w.clone().to("cpu").detach()

out_dict =  {"loss": loss_history, "x": x, "e": e, "ricci": ricci, 'riemann': riemann, 'w1': wald_1, 'w2': wald_2, 'w': w }
out_name = "./solution.mat" 
savemat(out_name, out_dict)
torch.save( tetradnet, "tetradnet")