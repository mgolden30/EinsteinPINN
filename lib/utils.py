''''
Put short functions here to keep the main training code as clean as possible
Let's say anything ~5 lines or less can be a "utils" function
'''

import torch
from scipy.io import savemat
import lib.model as model

def check_for_GPU():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def sample_uniform_cube(N):
    device = check_for_GPU()
    x = (2*torch.rand( (N, 4), requires_grad=True ) - 1).to(device)
    return x


def sample_normal_dist(N, std):
    device = check_for_GPU()
    x = torch.empty( (N,4) ).normal_(mean=0,std=std).to(device)
    x.requires_grad = True
    return x


def sample_black_holes(N, t_max, r_min, r_max, pos):
    '''
    PURPOSE: Generate data (t,x,y,z) within a uniform sphere with a hole cut out
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Uniformly sample a sphere with a hole cut out
    t = torch.rand((N, 1)) * t_max  # [0, t_max]
    r = torch.rand((N, 1)) * r_max
    cos_th = 2 * torch.rand((N, 1)) - 1
    phi = 2 * torch.pi * torch.rand((N, 1))

    sin_th = torch.sqrt(1 - cos_th ** 2)

    x = r * sin_th * torch.cos(phi)
    y = r * sin_th * torch.sin(phi)
    z = r * cos_th

    # Combine these to get (N,4) spacetime training points
    points = torch.cat((t, x, y, z), dim=1).to(device)
    
    pos = torch.tensor(pos).to(device)
    # Remove points too close to black holes
    for bh_pos in pos:
        dist = torch.norm(points[:, 1:] - bh_pos, dim=1)  # Compute distance to each black hole
        mask = dist >= r_min  # Mask points that are outside the cutoff radius
        points = points[mask]

    points.requires_grad = True
    return points



def save_network( tetradnet, x_train, x_test, loss_history, output_filename ):
    #save the network and diagnostic info we care about

    #Make a new PINN since there are no internal parameters outside of tetradnet
    pinn = model.EinsteinPINN( tetradnet )
    
    #save out the training and test data separately
    #training first
    x = x_train
    ricci, riemann, wald_1, wald_2, w = pinn.forward(x_train)
    ricci  = ricci.clone().to("cpu").detach()
    riemann= riemann.clone().to("cpu").detach()
    e      = tetradnet.forward(x).clone().to("cpu").detach()
    x      = x.clone().to("cpu").detach()
    wald_1 = wald_1.clone().to("cpu").detach()
    wald_2 = wald_2.clone().to("cpu").detach()
    w      = w.clone().to("cpu").detach()

    out_dict =  {"loss": loss_history, "x": x, "e": e, "ricci": ricci, 'riemann': riemann, 'w1': wald_1, 'w2': wald_2, 'w': w }
    matfile_train = output_filename + "_train.mat" 
    savemat(matfile_train, out_dict)

    #save out the training and test data separately
    #testing second
    x = x_test
    ricci, riemann, wald_1, wald_2, w = pinn.forward(x)
    ricci  = ricci.clone().to("cpu").detach()
    riemann= riemann.clone().to("cpu").detach()
    e      = tetradnet.forward(x).clone().to("cpu").detach()
    x      = x.clone().to("cpu").detach()
    wald_1 = wald_1.clone().to("cpu").detach()
    wald_2 = wald_2.clone().to("cpu").detach()
    w      = w.clone().to("cpu").detach()

    out_dict =  {"loss": loss_history, "x": x, "e": e, "ricci": ricci, 'riemann': riemann, 'w1': wald_1, 'w2': wald_2, 'w': w }
    matfile_test = output_filename + "_test.mat" 
    savemat(matfile_test, out_dict)

    #save out the network itself
    torch.save( tetradnet, output_filename + ".pth")