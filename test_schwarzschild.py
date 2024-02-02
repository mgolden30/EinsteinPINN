'''
PURPOSE:
Generate curvature tensors suing the trivial Schwarzschild tetrad. A matlab script checks for the inteded values out of Wald.
'''

import torch
from lib.model import SchwarzschildTetradNetwork, EinsteinPINN
import lib.utils as utils


device = utils.check_for_GPU()

#PUT ALL PARAMETERS HERE
epochs = 1 #number of training steps
learning_rate = 1e-3
num_training  = 1024 #number of training points

#Seed for reproducibility
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#Pick training data
x_train = utils.sample_uniform_cube( num_training ) #sample from [-1,1]^4
x_test  = utils.sample_uniform_cube( num_training ) #sample from [-1,1]^4

#shift r outside of event horizon
x_train[:,1] = x_train[:,1] + 3
x_test[ :,1] = x_test[ :,1] + 3

#Shift theta away from trouble
x_train[:,2] = x_train[:,2] + 2
x_test[ :,2] = x_test[ :,2] + 2

x_train = x_train.to(device)
x_test  = x_test.to(device)

#Create a new network for computing a random tetrad
tetradnet = SchwarzschildTetradNetwork().to(device)

#Just so we can use the save_network function
loss_history = torch.zeros( (epochs) )

#save everything out
#evaluation happens in the save_network functions
output_filename = "./network_output/schwarzschild_tetradnet" #utils.save_network will add file extensions for the appropriate output files.
utils.save_network( tetradnet, x_train, x_test, loss_history, output_filename )