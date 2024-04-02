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

class JetModel:
    def __init__( self, minR=0, maxR=0, minGamma=0, maxGamma=0, events_per_parampoint=10, two_prong=True, prong_pts=(1.,1.)):
        self.minR = minR
        self.maxR = maxR
        self.minGamma = minGamma
        self.maxGamma = maxGamma
        self.events_per_parampoint = events_per_parampoint
        self.two_prong = two_prong
        self.prong_pts = prong_pts
        # model: (exp(-R)+thetaR*exp(-R/alpha))^2 * (cos(y)*(1-thetaG) + sin(y)*thetaG)^2

        #alpha       = 2.
        #self.func   = ROOT.TF2("func", "(exp(-x)+({thetaR})*exp(-x/({alpha})))**2*(cos(y)*(1-{thetaG})+sin(y)*({thetaG}))**2".format( thetaR=self.thetaR, thetaG=self.thetaG, alpha=alpha) )
    
    def getEvents( self, Nevents ):
   
        #n1 = ctypes.c_double(0.) 
        #n2 = ctypes.c_double(0.) 
        RGamma = []
        for _ in range( 1+Nevents//self.events_per_parampoint ):
            #self.func.GetRandom2(n1, n2) 
            #RGamma.append( [n1.value, n2.value] )
            RGamma.append( [self.minR + (self.maxR-self.minR)*random.random(), self.minGamma + (self.maxGamma-self.minGamma)*random.random()] )

        truth = {'R':[], 'gamma':[]}

        pts, angles = [], []
        for iRGamma, (R, gamma) in enumerate(RGamma):
            n = self.events_per_parampoint if Nevents-self.events_per_parampoint*iRGamma>=self.events_per_parampoint else Nevents%self.events_per_parampoint
            if n==0: break
            pt, angle = sample( make_model( R=R, gamma=gamma, two_prong=self.two_prong, prong_pts=self.prong_pts), n )
            pts.append(torch.Tensor(pt))
            angles.append(torch.Tensor(angle))

            truth['R'].append( R )
            truth['gamma'].append( gamma )

        return torch.cat( tuple(pts), dim=0), torch.cat( tuple(angles), dim=0), None, torch.column_stack( tuple( torch.Tensor(truth[key]) for key in ['R', 'gamma']) )

    def __call__( self ):
        return 
