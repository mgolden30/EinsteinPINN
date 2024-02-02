''''
Here we can compile many different useful loss functions with different penalties and regularizers
'''

import torch
from lib.model import EinsteinPINN

def loss_basic( tetradnet, x ):
    #This is the most basic loss function I can think of. 
    #Set L equal to the mean(abs(ricci)) and divide by the mean(abs(riemann)) to encourage it to find some curvature 
    pinn = EinsteinPINN(tetradnet)

    # Forward pass to compute spacetime curvature
    ricci, riemann, _, _, _ = pinn.forward(x)

    err  = ricci/torch.mean( torch.abs(riemann), dim=(0,1,2,3,4) )

    return err



def loss_V1( tetradnet, x ):
    #compute the base error
    err1  = loss_basic(tetradnet,x)

    #add a local volume constraint abs(det(e)) - 1
    e = tetradnet.forward(x)
    err2  = torch.abs(torch.det(e)) - 1

    #reshape err1 [N,4,4] -> [N,16] and err2 [N] -> [N,1] so I can stack on dimension 1
    err1 = torch.reshape( err1, [-1,16] )
    err2 = torch.reshape( err2, [-1,1 ] )

    #stack into a single error
    err = torch.cat( (err1, err2), dim=1 )

    return err