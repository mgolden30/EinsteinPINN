import torch

import torch.optim as optim
from torch import nn
from lib.model import TetradNetwork_V1, EinsteinPINN
import lib.utils as utils

from lib.losses import loss_V1, loss_basic

device = utils.check_for_GPU()

epochs = 16
learning_rate = 1e-1
num_training = 1024

#For reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#Pick training data
x_train = utils.sample_uniform_cube( num_training ) #sample from [-1,1]^4
x_test  = utils.sample_uniform_cube( num_training ) #sample from [-1,1]^4

loss_history = torch.zeros( (epochs) )
tetradnet = torch.load("network_output/tetradnet_binary.pth")

criterion = nn.L1Loss()
optimizer = optim.LBFGS( tetradnet.parameters(), lr=1e-2)

def closure():
    optimizer.zero_grad()  # Clear gradients
    err = loss_basic(tetradnet, x_train)
    loss = criterion(err, torch.zeros_like(err))
    loss.backward(retain_graph=True)   # compute gradients
    return loss

for epoch in range(epochs):
    # Forward pass
    err  = loss_basic( tetradnet, x_train )

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

output_filename = "./network_output/tetradnet_finetuned" #utils.save_network will add file extensions for the appropriate output files.
utils.save_network( tetradnet, x_train, x_test, loss_history, output_filename )