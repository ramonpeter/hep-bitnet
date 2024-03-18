import numpy as np

# def Etot_Einc_discrete(hlf):
#     target_energies = 2**np.linspace(8, 23, 16)
#     which_showers = ((hlf.Einc.squeeze() >= target_energies[i]) & (hlf.Einc.squeeze() < target_energies[i+1])).squeeze()
    
#     return hlf.GetEtot()[which_showers] / hlf.Einc.squeeze()[which_showers]

def Etot_Einc(hlf):
    return hlf.GetEtot() / hlf.Einc.squeeze()

def E_layers(hlf, layer):
    return hlf.GetElayers()[layer]

def ECEtas(hlf, layer):
    return hlf.GetECEtas()[layer]
    
def ECPhis(hlf, layer):
    return hlf.GetECPhis()[layer]

def ECWidthEtas(hlf, layer):
    return hlf.GetWidthEtas()[layer]

def ECWidthPhis(hlf, layer):
    return hlf.GetWidthPhis()[layer]

def cell_dist(hlf):
    return hlf.shower.flatten()

def sparsity(hlf, layer):
    return hlf.GetSparsity()[layer]