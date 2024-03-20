import ctypes
import ROOT
from MicroMC import sample, make_model
import torch
import random
import math

if __name__=="__main__":
    import sys
    sys.path.insert(0, '..')
    sys.path.insert(0, '../..')

class EFTModel:
    def __init__( self, thetaR=0, thetaG=0, events_per_parampoint=10):
        self.thetaR = thetaR
        self.thetaG = thetaG

        self.events_per_parampoint = events_per_parampoint

        # model: (exp(-R)+thetaR*exp(-R/alpha))^2 * (cos(y)*(1-thetaG) + sin(y)*thetaG)^2

        alpha       = 2.
        self.func               = ROOT.TF2("func", "(exp(-x)+({thetaR})*exp(-x/({alpha})))**2*(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))**2".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
        self.func_thetaR        = ROOT.TF2("func", "2*(exp(-x)+({thetaR})*exp(-x/({alpha})))*exp(-x/({alpha}))*(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))**2".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
        self.func_thetaR_thetaR = ROOT.TF2("func", "2*exp(-2*x/({alpha})) *(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))**2".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
        self.func_thetaG        = ROOT.TF2("func", "2*(exp(-x)+({thetaR})*exp(-x/({alpha})))**2*(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))*(sin(y)-cos(y))".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
        self.func_thetaG_thetaG = ROOT.TF2("func", "2*(exp(-x)+({thetaR})*exp(-x/({alpha})))**2*(sin(y)-cos(y))**2".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
        self.func_thetaR_thetaG = ROOT.TF2("func", "4*(exp(-x)+({thetaR})*exp(-x/({alpha})))*exp(-x/({alpha}))*(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))*(sin(y)-cos(y))".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )

        self.derivatives = [ tuple(), ('thetaR',), ('thetaR','thetaR') ]
    
    def getEvents( self, Nevents ):
   
        n1 = ctypes.c_double(0.) 
        n2 = ctypes.c_double(0.) 
        RGamma = []
        for _ in range( 1+Nevents//self.events_per_parampoint ):
            self.func.GetRandom2(n1, n2) 
            RGamma.append( [n1.value, n2.value] )

        weights = {tuple():[], ('thetaR',):[], ('thetaG',):[], ('thetaR','thetaR'):[], ('thetaG', 'thetaG'):[], ('thetaG', 'thetaR'):[]}
        truth = {'R':[], 'gamma':[]}

        pts, angles = [], []
        for iRGamma, (R, gamma) in enumerate(RGamma):
            n = self.events_per_parampoint if Nevents-self.events_per_parampoint*iRGamma>=self.events_per_parampoint else Nevents%self.events_per_parampoint
            if n==0: break
            pt, angle = sample( make_model( R=R, gamma=gamma ), n )
            pts.append(torch.Tensor(pt))
            angles.append(torch.Tensor(angle))

            weights[tuple()].extend( [self.func.Eval(R, gamma)]*n )
            weights[('thetaR',)].extend( [self.func_thetaR.Eval(R, gamma)]*n )
            weights[('thetaR','thetaR')].extend( [self.func_thetaR_thetaR.Eval(R, gamma)]*n )
            #weights[('thetaG',)].extend( [self.func_thetaG.Eval(R, gamma)]*n )
            #weights[('thetaG', 'thetaG')].extend( [self.func_thetaG_thetaG.Eval(R, gamma)]*n )
            #weights[('thetaG', 'thetaR')].extend( [self.func_thetaR_thetaG.Eval(R, gamma)]*n )
            
            truth['R'].append( R )
            truth['gamma'].append( gamma )

        #weights = {key:torch.Tensor(val) for key, val in weights.items()}
        weights = {key:torch.Tensor(weights[key])/torch.Tensor(weights[tuple()]) for key in self.derivatives}
        weights = torch.column_stack( tuple( torch.Tensor(weights[key]) for key in self.derivatives) )
        truth   = torch.column_stack( tuple( torch.Tensor(truth[key]) for key in ['R', 'gamma']) )

        return torch.cat( tuple(pts), dim=0), torch.cat( tuple(angles), dim=0), weights, truth 

    def __call__( self ):
        return 

