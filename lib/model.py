import torch as torch
import torch.nn as nn
import torch.nn.init as init

import lib.utils as utils

device = utils.check_for_GPU()




class TetradNetwork_V1(nn.Module):
    def __init__(self, num_layers, feature_size):
        '''
        Create a network that goes from 4 -> L -> L -> ... -> L -> 16, where L is the feature_size and there are num_layers number of layers.
        '''

        super().__init__()
        
        self.layer1 = nn.Linear(4, feature_size).to(device)
        self.hidden_layers = nn.ModuleList([nn.Linear(feature_size, feature_size).to(device) for _ in range(num_layers)])
        self.layer_out = nn.Linear(feature_size, 16).to(device)
        
        self._init_weights()

    def forward(self, x):
        x = self.layer1(x)
        for layer in self.hidden_layers:
            x = layer(x) #apply linear layer
            x = torch.cos(x) #activation function
            #x = torch.exp( - torch.square(x) )
        x = self.layer_out(x)
        #reshape to 4x4 matrices
        x = torch.reshape( x, [-1, 4, 4] )
        return x

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)





class TetradNetwork_V2(nn.Module):
    '''
    PURPOSE:
    Trigonometric Network (https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=eed78c3d057f9be2587c4f6a5e68956974bb5a26)
    "Iterative Improvement of Trigonometric Networks"
    '''
    def __init__(self):
        super().__init__()
        L = 16
        self.layer1  = nn.Linear(    4,   L ).to(device)
        self.layer2  = nn.Linear(    2*L, L ).to(device)
        self.layer3  = nn.Linear(    2*L, 16).to(device)
        self._init_weights()

    def forward(self, x):
        #forward pass of the neural network
        x = self.layer1(x)
        x = torch.cat( (torch.sin(x), torch.cos(x)), dim=1 )
        
        x = self.layer2(x)
        x = torch.cat( (torch.sin(x), torch.cos(x)), dim=1 )
        
        x = self.layer3(x)
        x = torch.reshape(x, [-1,4,4])
        
        return x
    
    def _init_weights(self):
        # Initialize weights for layer1
        init.xavier_uniform_(self.layer1.weight)
        init.constant_(self.layer1.bias, 0.0)

        # Initialize weights for layer2
        init.xavier_uniform_(self.layer2.weight)
        init.constant_(self.layer3.bias, 0.0)
        
        # Initialize weights for layer3
        init.xavier_uniform_(self.layer3.weight)
        init.constant_(self.layer3.bias, 0.0)
 

class TetradNetwork_V3(nn.Module):
    '''
    PURPOSE:
    Trigonometric Network (https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=eed78c3d057f9be2587c4f6a5e68956974bb5a26)
    "Iterative Improvement of Trigonometric Networks"
    '''
    def __init__(self):
        super().__init__()
        L = 8 #number of thetas computed each layer
        self.layer1  = nn.Linear(      4, L ).to(device) 
        self.layer2  = nn.Linear(    2*L, L ).to(device)
        self.layer3  = nn.Linear(    4*L, 16).to(device)
        self._init_weights()

    def forward(self, x):
        #forward pass of the neural network
        x = self.layer1(x)
        x = torch.cat( (torch.sin(x), torch.cos(x)), dim=1 )
        
        x0 = x.clone()
        x = self.layer2(x)
        x = torch.cat( (x0, torch.sin(x), torch.cos(x)), dim=1 )

        x = self.layer3(x)
        x = torch.reshape(x, [-1,4,4])
        
        return x
    
    def _init_weights(self):
        # Initialize weights for layer1
        init.xavier_uniform_(self.layer1.weight)
        init.constant_(self.layer1.bias, 0.0)

        # Initialize weights for layer2
        init.constant_(self.layer2.weight, 0.0)
        init.constant_(self.layer3.bias, 0.0)
        
        # Initialize weights for layer3
        init.xavier_uniform_(self.layer3.weight)
        init.constant_(self.layer3.bias, 0.0)



class SchwarzschildTetradNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        t = x[:,0]
        r = x[:,1]
        th= x[:,2]
        ph= x[:,3]

        N = x.shape[0]
        e = torch.zeros( (N,4,4) ).to(device)

        #Standard diagonal tetrad
        e[:,0,0] =     torch.sqrt( 1.0 - 2.0/r )
        e[:,1,1] = 1.0/torch.sqrt( 1.0 - 2.0/r )
        e[:,2,2] = r
        e[:,3,3] = r*torch.sin(th)
        return e

class MultiBlackHoleNetwork(nn.Module):
    def __init__(self, pos):
        super().__init__()
        self.pos = pos

    def forward(self, x):
        N = x.shape[0]
        e = torch.zeros( (N,4,4) ).to(device)

        M = 1.0 #masses of all black holes taken to be unity
        psi = 1 # psi = 1 + sum(M/2r)
        for bh_pos in self.pos:
            r = torch.norm( x[:, 1:] - bh_pos, dim=1)  # Compute distance to each black hole
            psi = psi + M/2/r

        #See if a different slicing induces dynamics
        e[:,0,0] = 1
        e[:,1,1] = torch.square(psi)
        e[:,2,2] = torch.square(psi)
        e[:,3,3] = torch.square(psi)
        return e




class EinsteinPINN(nn.Module):
    def __init__(self, tetradnetwork):
        super().__init__()
        self.tetradnetwork = tetradnetwork        
        
    def forward(self, x):
        '''
        PURPOSE:
        Evaluate the Ricci curvature defined by a tetrad.

        DETAILS:
        I am going to attempt use einsum and only einsum where possible for maximal readability.
        I will use the convention of Greek indices \mu,\nu for spatial tangent space DOF and capital Latin IJ 
        for Minkowski indices. For einsum, we need single letter indices, so have \mu and \nu correspond to m, n. 
        '''


        # Step 1: evaluate trivial functions of tetrad e_{\mu I}
        e     = self.tetradnetwork(x)     #evaluate the tetradfunction e_{\mu I}
        de    = self.tetrad_gradient(e,x) #take the partial derivatives of tetrad components
        e_inv = torch.inverse( e ) #e_{\mu I} -> e^{I \mu}
        e_inv = torch.einsum( "bIm->bmI", e_inv ) #Switch index order to e^{\mu I}, same order as e_{\mu I}
        minko = torch.diag( torch.tensor([-1.0, 1.0, 1.0, 1.0]) ).to(device) #Minkowski metric
        
        e_mat = torch.einsum( 'bmJ,IJ->bmI', e_inv, minko ) # e^\mu_I (inverse with Mink index lowered)

        # Step 2: compute the connection one-forms
        rotation = torch.einsum( "bmI,bnJ,bmnK->bIJK", e_mat, e_mat, de ) #psuedo-rotation coefficients (cov derivative replaced w partial)
        tau      = rotation - torch.einsum( "bIJK->bJIK", rotation ) #antisymmetric in first two indices
        
        # For some reason my code works if the construction takes 3 steps
        # rotation = ( tau + torch.einsum("bIJK->bKJI", tau) + torch.einsum("bIJK->bKIJ", tau0) )/2.0 #compute the true rotation coeffs
        rotation = torch.einsum("bIJK->bKJI", tau)
        rotation = rotation + (tau - torch.einsum("bIJK->bIKJ", tau))
        rotation = rotation/2.0
        
        w        = torch.einsum( "bKIJ,bmL,LK->bmIJ", rotation, e, minko ) #w_{\mu IJ}
        w2       = torch.einsum( "bmIK,KJ->bmIJ", w, minko ) # w_{\mu I}^J

        #Optionally check the equations of structure to see if our one-form agrees with Wald
        wald_1, wald_2 = self.check_wald( de, e, w2 )

        #Step 3: Riemann curvature and Ricci curvature
        dw2     = self.connection_gradient( w2, x ) # \partial_\mu w_{\nu I}^J
        riemann = dw2 + torch.einsum( "bmik,bnkj->bmnij", w2, w2 ) #R_{\mu\nu I}^J (before symm)
        riemann = riemann - torch.einsum( "bmnIJ->bnmIJ", riemann) #R_{\mu\nu I}^J
        ricci   = torch.einsum( "bmnij,bnj->bmi", riemann, e_mat ) #R_{\mu I} = R_{\mu\nu I}^J e^\nu_J
        ricci   = torch.einsum( "bmj,bmi->bij", ricci, e_mat)      #R_{IJ} = R_{\mu J} e^\mu_I

        #Might as well cast the Riemann tensor full in Minkwoski space, fully covariant
        riemann = torch.einsum( "bmnIJ,bmK,bnL,JM->bKLIM", riemann, e_mat, e_mat, minko)

        return ricci, riemann, wald_1, wald_2, w
    
    def tetrad_gradient(self, e, x ):
        my_tup = [0] * 4 * 4 
        for i in range(4):
            for j in range(4):
                my_tup[4*i+j] = torch.autograd.grad( e[:,i,j], x, grad_outputs=torch.ones_like(e[:,i,j]), create_graph=True, allow_unused=True )[0]
        
        de = torch.cat( my_tup, dim=-1 )
        de = torch.reshape( de, [-1, 4, 4, 4] )
        #Move the partial derivative index from the back to the front
        de = torch.einsum( "bnIm->bmnI", de ) 
        return de

    def connection_gradient(self, w2, x ):
        #Compute derivatives of connection one-forms
        my_tup = [0] * 4 * 4 * 4
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    my_tup[ 4*4*i + 4*j + k ] = torch.autograd.grad( w2[:,i,j,k], x, grad_outputs=torch.ones_like(w2[:,i,j,k]), create_graph=True, allow_unused=True )[0]     
        dw2 = torch.cat( my_tup, dim=-1 )
        dw2 = torch.reshape( dw2, [-1, 4, 4, 4, 4] )
        dw2 = torch.einsum( "bnIJm->bmnIJ", dw2 ) 
        return dw2

    def check_wald(self, de, e, w2):
        # To check that w is calculated right, let's compute the equations of structure
        # Wald p 52, eq 3.4.27
        wald_1 = de
        wald_2 = torch.einsum( "bmJ,bnIJ->bmnI", e, w2 )

        #antisymmetrize both
        wald_1 = wald_1 - torch.einsum( "bmnI->bnmI", wald_1 )
        wald_2 = wald_2 - torch.einsum( "bmnI->bnmI", wald_2 )
        return wald_1, wald_2