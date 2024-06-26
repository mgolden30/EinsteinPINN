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

    #compute the mean abs(Riemannian curvature)
    mean_Riemann = torch.mean( torch.abs(riemann), dim=(1,2,3,4) )
    mean_Ricci   = torch.mean( torch.abs(ricci),   dim=(1,2) )
    #add two costs: one for Ricci = 0 and one for <|Riemann|> = 1 so we get non-flat spacetime
    err = mean_Ricci / mean_Riemann
    
    #scale it by det(e)^2 so the sum is coordinate invariant
    #e   = tetradnet.forward(x)
    #err = err * torch.square(torch.det(e))
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


def loss_V2( tetradnet, x ):
    pinn = EinsteinPINN(tetradnet)

    # Forward pass to compute spacetime curvature
    ricci, riemann, _, _, _ = pinn.forward(x)
    
    #Instead of a direct ratio, multiply by 1 + 1/Riemann
    regularization = 1.0 + 1.0/torch.mean( torch.abs(riemann), dim=(0,1,2,3,4) ) 
    err = ricci * regularization
    return err

def loss_black_hole( tetradnet, x, schwarzschildnet, x0 ):
    '''
    PURPOSE:
    Construct a physics-informed loss for single black hole data
    '''

    #Check if the EFEs hold
    err1 = loss_basic( tetradnet, x )

    #Check if we reproduce a Schwarzschild tetrad at t=0
    #x0.requires_gradient = True #so we can take time derivative
    x0 = torch.tensor(x0, requires_grad=True)
    e0        = tetradnet.forward(x0)
    e0_target = schwarzschildnet.forward(x0)
    err2      = torch.abs(e0 - e0_target)

    #Enforce time derivative of 
    e0_grad = torch.autograd.grad(e0, x0, grad_outputs=torch.ones_like(e0), create_graph=True)[0]
    err3 = torch.abs( e0_grad[:, 3] )  # Assuming x0[:,3] corresponds to the variable of interest

    err1 = torch.reshape( err1, [-1] )
    err2 = torch.reshape( err2, [-1] )
    err3 = torch.reshape( err3, [-1] )  
    
    #stack into a single error
    err = torch.cat( (err1, err2, err3), dim=0 )
    
    return err