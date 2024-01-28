import torch as torch
import torch.nn as nn

class TetradNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        L = 32
        self.layer1  = nn.Linear(  4, L )
        self.layer2  = nn.Linear(  L, L )
        self.layer3  = nn.Linear(  L, L )
        self.layer4  = nn.Linear(  L, 16)

    def forward(self, x):
        A = 1 #amplitude
        
        x0 = torch.clone(x)
        x = self.layer1(x)
        #x = A*torch.exp(-torch.square(x))
        x =  A*torch.sinc(x)
        #x = torch.cat( (x,x0), dim=1 )
        
        x = self.layer2(x)
        #x = A*torch.exp(-torch.square(x))
        x = A*torch.sinc(x)
        #x = torch.cat( (x,x0), dim=1 )
        
        x = self.layer3(x)
        #x = A*torch.tanh(x)
        #x = A*torch.exp(-torch.square(x))
        x = A*torch.sinc(x)

        x = self.layer4(x)
        x = torch.reshape(x, [-1,4,4])
        
        #x = torch.linalg.matrix_exp(x)
        return x


class SchwarzschildTetradNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        L = 32
        self.layer1  = nn.Linear(  4, L )
        self.layer2  = nn.Linear(  L, L ) 
        self.layer3  = nn.Linear(  L, 16)


    def forward(self, x):
        r = x[:,1]
        th= x[:,2]

        N = x.shape[0]
        e = torch.zeros( (N,4,4) )
        e[:,0,0] =     torch.sqrt( 1-2/r )
        e[:,1,1] = 1.0/torch.sqrt( 1-2/r )
        e[:,2,2] = r
        e[:,3,3] = r*torch.sin(th)
        return e

class FlatTetradNetwork(nn.Module):
    #define a flat spacetime tetrad with nonstandard coordinates
    #t = t'^3 and so on
    def __init__(self):
        super().__init__()
        L = 32
        self.layer1  = nn.Linear(  4, L )
        self.layer2  = nn.Linear(  L, L ) 
        self.layer3  = nn.Linear(  L, 16)


    def forward(self, x):
        N = x.shape[0]
        e = torch.zeros( (N,4,4) )
        for i in range(4):
            e[:,i,i] = 3*x[:,i]*x[:,i] #from differentiating x^3
        return e

class EinsteinPINN(nn.Module):
    def __init__(self, tetradnetwork):
        super().__init__()
        self.tetradnetwork = tetradnetwork        
        
    def forward(self, x):
        e = self.tetradnetwork(x) #evaluate the tetradfunction
        
        #autodiff the streamfunction
        #print( e.shape )
        my_tup = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
        for i in range(4):
            for j in range(4):
                my_tup[4*i+j] = torch.autograd.grad( e[:,i,j], x, grad_outputs=torch.ones_like(e[:,i,j]), create_graph=True, allow_unused=True )[0]
        
        de = torch.cat( my_tup, dim=-1 )
        de = torch.reshape( de, [-1, 4, 4, 4] )
        #Assume the index order is [N \nu I \mu] of \partial_\mu e_{\nu I}
        #I hate this, so we should permute the indices

        de = torch.permute( de, [0,3,1,2] ) 

        #Note taking the inverse will inadavertently change the order of indices to [N, I, \mu]
        e_inv = torch.inverse( e )
        e_inv = torch.permute( e_inv, [0,2,1] ) #Now it is [N, \mu ,I], just like e before inversion
        
        #define the Minkowski metric of size [4,4]
        minkowski = torch.diag( torch.tensor([-1.0, 1.0, 1.0, 1.0]) ) 

        #Lower the Mink index of inverse: e^\mu_I = e^{\mu J} \eta_{JI}
        e_inv0 = torch.clone(e_inv)
        e_inv  = torch.einsum( 'bmj,ji->bmi', e_inv, minkowski )

        # define psuedo-one-form pw_{\mu I J} = e^\nu_I \partial_\mu e_{\nu J} 
        pw = torch.einsum( "bni,bmnj->bmij", e_inv, de )
        
        #compute psuedo-rotation coefficients pr_{KIJ} = e^\mu_K pw_{\mu IJ}
        pr = torch.einsum( "bmij,bmk->bkij", pw, e_inv )

        #antisymmetrize tau <- pr_{KIJ} - pr_{IKJ} = same with r instead of pr
        tau = pr - torch.permute(pr, [0,2,1,3])

        #compute the Ricci rotation coefficients r_{IJK}
        r = (tau + torch.permute( tau, [0,3,2,1] ) + torch.permute( tau, [0,3,1,2] ))/2

        #Take the first index back to cotangent space
        r = torch.einsum( "bljk,li->bijk", r, minkowski ) #raise first index r^I_{JK}
        w = torch.einsum( "bijk,bmi->bmjk", r, e ) #apply e_{\mu I} r^I{}_{JK}

        #Compute derivatives of connection one-forms
        my_tup = [0] * 4 * 4 * 4
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    my_tup[ 4*4*i + 4*j + k ] = torch.autograd.grad( w[:,i,j,k], x, grad_outputs=torch.ones_like(w[:,i,j,k]), create_graph=True, allow_unused=True )[0]     
        dw = torch.cat( my_tup, dim=-1 )
        dw = torch.reshape( dw, [-1, 4, 4, 4, 4] )
        # Assume we need to do the same thing as before the bring the partial derivative to the front
        dw = torch.permute( dw, [0,4,1,2,3] ) #{N, \mu, \nu ,I ,J} of \partial_\mu w_{\nu I J}

        #create a connection one-form with last index raised
        w2 = torch.einsum( "bmik,kj->bmij", w, minkowski ) # w_{\mu I}^J

        #compute the Riemann tensor
        riemann = dw - torch.permute(dw,[0,2,1,3,4]) \
                     + torch.einsum( "bmik,bnkj->", w2, w ) \
                     - torch.einsum( "bnik,bmkj->", w2, w )
        
        #print( riemann.shape )

        #compute Ricci tensor by tracing with e_inv
        ricci = torch.einsum( "bmnij,bnj->bmi", riemann, e_inv0 ) # use e^{\mu I}

        #Right now Riemann is in form R_{\mu\nu IJ} and Ricci is R_{\mu I}
        #Let's put everything in Minkowski space so its magnitude is meaningful
        ricci = torch.einsum( "bmj,bmi->bij", ricci, e_inv)

        riemann = torch.einsum( "bmnij,bmk->bknij", riemann, e_inv )
        riemann = torch.einsum( "bknij,bnl->bklij", riemann, e_inv )

        return ricci, riemann
       
        