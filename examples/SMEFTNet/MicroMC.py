import torch
import torch.distributions as D
import numpy as np
from math import pi, sin, cos
import ROOT

def make_model( R = 1, gamma = 0, var = 0.3, two_prong = True, prong_pts=(1.,1.)):
    if two_prong:
        mix  = D.Categorical(torch.Tensor(prong_pts))
        comp = D.Independent(D.Normal(
                    torch.Tensor(((R*cos(gamma), R*sin(gamma)), (-R*cos(gamma), -R*sin(gamma)))),
                    torch.Tensor(((var,var),(var,var)))), 1)
        return D.MixtureSameFamily(mix, comp)
    else:
        comp = D.Independent(D.Normal(
                    torch.Tensor(((R*cos(gamma), R*sin(gamma)))),
                    torch.Tensor(((var,var)))),1)
        return comp 

def make_TH2(R = 1, gamma = 0, var = 0.3, two_prong = True):
    if two_prong:
        return ROOT.TF2("f",".5/(2*pi*({var}))*(exp(-.5*(x+({R})*cos({gamma}))**2/({var})**2 - .5*(y+({R})*sin({gamma}))**2/({var})**2) + exp(-.5*(x-({R})*cos({gamma}))**2/({var})**2 - .5*(y-({R})*sin({gamma}))**2/({var})**2))".format(var=var,R=R,gamma=gamma),-2,2,-2,2)
    else:
        return ROOT.TF2("f","1./(2*pi*({var}))*exp(-.5*(x-({R})*cos({gamma}))**2/({var})**2 - .5*(y-({R})*sin({gamma}))**2/({var})**2)".format(var=var,R=R,gamma=gamma),-2,2,-2,2)

mean_Nparticles = 50
Nparticle_pad   = 80
pt_jet = 100
smearing=0.1

def sample( model, Nevents ):

    Nparticles = D.Poisson(mean_Nparticles)
    angles  = torch.Tensor([torch.nn.functional.pad( model.sample((Nparticles.sample().int(),)), (0,0,0,Nparticle_pad))[:Nparticle_pad].numpy() for _ in range( Nevents )])
    mask    = angles.abs().sum(dim=-1)!=0
    pt      = torch.exp(model.log_prob(angles))
    pt      = pt_jet*torch.distributions.log_normal.LogNormal(0,.2).sample((Nevents,Nparticle_pad))*(pt*mask)/(pt*mask).sum(dim=-1).reshape(-1,1)
    return pt.numpy(), angles.numpy()

def mix_model( model1, model2, Nevents ):

    Nparticles = D.Poisson(mean_Nparticles/2.)
    npart = [ (Nparticles.sample().int().item(), Nparticles.sample().int().item()) for _ in range(Nevents)]
    #print ( [torch.cat( (torch.ones(npart1), -1*torch.ones(npart2)), dim=0) for npart1, npart2 in npart] )
    pop     = torch.Tensor([ torch.cat( (torch.ones(npart1), -1*torch.ones(npart2), torch.zeros(max(0,Nparticle_pad-npart1-npart2))), dim=0)[:Nparticle_pad].numpy()  for npart1, npart2 in npart])
    angles  = torch.Tensor([torch.nn.functional.pad( torch.cat( (model1.sample((npart1,)), model2.sample((npart2,))), dim=0), (0,0,0,Nparticle_pad))[:Nparticle_pad].numpy() for npart1, npart2 in npart])
    #print ("", angles.shape, model1.log_prob(angles))
    pt      = torch.exp(model1.log_prob(angles)) + torch.exp(model2.log_prob(angles))
    mask    = angles.abs().sum(dim=-1)!=0
    #print ("mask", mask.shape, mask)
    pt      = pt_jet*torch.distributions.log_normal.LogNormal(0,.2).sample((Nevents,Nparticle_pad))*(pt*mask)/(pt*mask).sum(dim=-1).reshape(-1,1)

    return pt.numpy(), angles.numpy(), pop

def getModels( models ):
    if models == 'R1dGamma':
        signal     = make_model( R=1, gamma=0, var=0.3 )
        background = make_model( R=1, gamma=pi/2, var=0.3 )
    elif models == 'R0vsR1':
        signal     = make_model( R=1, gamma=0, var=0.3 )
        background = make_model( R=0, gamma=0, var=0.3 )
    else:
        raise NotImplementedError

    return signal, background

from sklearn.model_selection import train_test_split
def getEvents( signal, background, nTraining, test_size=None, train_size=None, ret_sig_bkg=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pt_sig, angles_sig = sample(signal, nTraining)
    pt_bkg, angles_bkg = sample(background, nTraining)

    if ret_sig_bkg:
        return torch.Tensor(pt_sig).to(device), torch.Tensor(angles_sig).to(device), torch.Tensor(pt_bkg).to(device), torch.Tensor(angles_bkg).to(device)

    label_sig = torch.ones(  len(pt_sig) )
    label_bkg = torch.zeros( len(pt_bkg) )

    return train_test_split( 
        torch.Tensor(np.concatenate( (pt_sig, pt_bkg) )).to(device), 
        torch.Tensor(np.concatenate( (angles_sig, angles_bkg) )).to(device), 
        torch.Tensor(np.concatenate( (label_sig, label_bkg) )).to(device),
        test_size=test_size, train_size=train_size
    )

