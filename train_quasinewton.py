import numpy as np
import os
import torch
import random

import torch.optim as optim
from torch import nn
from lib.model import TetradNetwork, SchwarzschildTetradNetwork, EinsteinPINN

from scipy.io import savemat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_folder = "network_output/"

epochs = 64*3
learning_rate = 1e-3

#For reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

N = 1024
x = (2*torch.rand( (N, 4), requires_grad=True ) - 1).to(device)

'''
# number of boundary points
Nb = 1024
xb = (2*torch.rand( (N, 4), requires_grad=True ) - 1)
random.seed(1)
for p in range(Nb):
    j = random.randrange(4)
    xb[p,j] = 2*random.randrange(0,2)-1 #-1 or 1
'''
e_mean = torch.zeros( (4,4) )
e_mean[0,0] = 1
e_mean[1,1] = 1
e_mean[2,2] = 1
e_mean[3,3] = 1

e_mean = e_mean.to(device)

#for testing Schwarzschild
#x = 2.1 + 0.5*torch.rand( (N, 4), requires_grad=True ) 
#tetradnet = SchwarzschildTetradNetwork()

#tetradnet = TetradNetwork().to(device)
tetradnet = torch.load("tetradnet_1.pth")
pinn      = EinsteinPINN(tetradnet).to(device)

criterion = nn.L1Loss()
#optimizer = optim.Adam( pinn.parameters(), lr=learning_rate )

optimizer = optim.LBFGS( pinn.parameters(), lr=1e-2)


def closure():
    optimizer.zero_grad()  # Clear gradients
    ricci, riemann, _, _, _ = pinn.forward(x)
    err_einstein = ricci/torch.mean( torch.abs(riemann), dim=(0,1,2,3,4) )
    #I want to enforce that the spacetime volume is 8 in total
    e = tetradnet.forward(x)

    #volume is sqrt(-g) = abs(det(e))
    #err_volume = torch.mean( torch.abs(torch.linalg.det(e)) - 1)

    #Learn a regularization to promote approxiamtely Minkowski coordinates
    #e = tetradnet.forward(x)
    #g = torch.einsum( "bmI,bnJ,IJ->bmn", e, e, minkowski )    
    #deviation = (torch.mean(g,dim=0) - minkowski) #want mean over position to be approximately identity
    #reg = torch.mean(torch.abs(deviation), dim=(0,1))

    err = err_einstein
    #err = torch.cat( (torch.reshape(err_einstein, [-1]), torch.reshape( N*err_volume, [-1])) )
    loss = criterion(err, torch.zeros_like(err))
    #loss.backward()  # Backward pass
    loss.backward(retain_graph=True)   # compute gradients
    return loss

epochs = 100
loss_history = torch.zeros( (epochs) )
for epoch in range(epochs):
    #generate new training data
    #xs = generate_samples(n)

    # Forward pass
    ricci, riemann, _, _, _ = pinn.forward(x)
    err_einstein = ricci/torch.mean( torch.abs(riemann), dim=(0,1,2,3,4) )
    #I want to enforce that the spacetime volume is 8 in total
    e = tetradnet.forward(x)

    #volume is sqrt(-g) = abs(det(e))
    #err_volume = torch.mean( torch.abs(torch.linalg.det(e)) - 1)

    #Learn a regularization to promote approxiamtely Minkowski coordinates
    #e = tetradnet.forward(x)
    #g = torch.einsum( "bmI,bnJ,IJ->bmn", e, e, minkowski )    
    #deviation = (torch.mean(g,dim=0) - minkowski) #want mean over position to be approximately identity
    #reg = torch.mean(torch.abs(deviation), dim=(0,1))

    err = err_einstein
    #err = torch.cat( (torch.reshape(err_einstein, [-1]), torch.reshape( N*err_volume, [-1])) )

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


'''
loss_history = torch.zeros( (epochs) )

minkowski = torch.diag( torch.tensor([-1.0,1.0,1.0,1.0]) ).to(device)

for epoch in range(epochs):
    # Forward pass (with output of PickDomains!)
    ricci, riemann, _, _, _ = pinn.forward(x)

    #We want the Ricci tensor to vanish. I am weighting by the Riemann tensor
    err_einstein = ricci/torch.mean( torch.abs(riemann), dim=(0,1,2,3,4) )

    #Learn a regularization to promote approxiamtely Minkowski coordinates
    #e = tetradnet.forward(x)
    #g = torch.einsum( "bmI,bnJ,IJ->bmn", e, e, minkowski )    
    #deviation = (torch.mean(g,dim=0) - minkowski) #want mean over position to be approximately identity
    #reg = torch.mean(torch.abs(deviation), dim=(0,1))

    #err = err_einstein *( 1 + reg)
    #err = torch.cat( (torch.reshape(err_einstein, [-1]), torch.reshape( 16*N*reg, [-1])) )
    #print(err.shape)
    err = err_einstein

    # Compute the MSE loss
    loss = criterion(err, torch.zeros_like(err))  # assuming you want to minimize pinn.forward(xs) to zero
    loss_history[epoch] = loss.detach()

    # clear previous gradients
    optimizer.zero_grad()
    loss.backward(retain_graph=True)   
        
    print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")
    
    # update model parameters
    optimizer.step()

    if( epoch % 64 == 0 ):
        #reduce learning rate
        learning_rate = learning_rate/2
        optimizer = optim.Adam( pinn.parameters(), lr=learning_rate )
'''


ricci, riemann, wald_1, wald_2, w = pinn.forward(x)

minkowski = torch.diag( torch.tensor([-1.0,1.0,1.0,1.0]) ).to(device)

scalar  = torch.einsum( "bIJKL,bIJKL,II,JJ,KK,LL->b", riemann, riemann, minkowski, minkowski, minkowski, minkowski )
scalar2 = torch.einsum( "bIJ,bIJ,II,JJ->b", ricci, ricci, minkowski, minkowski )
ricci = ricci.clone().to("cpu").detach()
riemann= riemann.clone().to("cpu").detach()
e     = tetradnet.forward(x).clone().to("cpu").detach()
x     = x.clone().to("cpu").detach()
wald_1= wald_1.clone().to("cpu").detach()
wald_2= wald_2.clone().to("cpu").detach()
w     = w.clone().to("cpu").detach()
scalar=scalar.clone().to("cpu").detach()
scalar2=scalar2.clone().to("cpu").detach()

out_dict =  {"loss": loss_history, "x": x, "e": e, "ricci": ricci, 'riemann': riemann, 'w1': wald_1, 'w2': wald_2, 'w': w, "scalar": scalar, "scalar2": scalar2 }
out_name = "./solution.mat" 
savemat(out_name, out_dict)
torch.save( tetradnet, "tetradnet_1.pth")