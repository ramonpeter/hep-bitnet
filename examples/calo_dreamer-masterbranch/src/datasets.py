import os
import torch
import numpy as np
import gc

from torch.utils.data import Dataset, DataLoader
from Util.util import *
from transforms import *

def get_loaders(hdf5_file, particle_type, xml_filename, val_frac, batch_size,
                transforms, eps=1.e-10, device='cpu', shuffle=True, width_noise=0.0,
                single_energy=None, aug_transforms=False):

    print("Dict of preprocessing is: ")
    print(transforms)

    train_dataset = CaloChallengeDataset(hdf5_file, particle_type, xml_filename, 
                    val_frac=val_frac, transform=transforms, split='training', device=device,
                    single_energy=single_energy, aug_transforms=aug_transforms)
    
    
    val_dataset = CaloChallengeDataset(hdf5_file, particle_type, xml_filename,
                    val_frac=val_frac, transform=transforms, split='validation', device=device,
                    single_energy=single_energy, aug_transforms=aug_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, val_dataloader, train_dataset.layer_boundaries


class CaloChallengeDataset(Dataset):
    """ Dataset for CaloChallenge showers """
    def __init__(self, hdf5_file, particle_type, xml_filename, val_frac=0.3, 
            transform=None, split=None, device='cpu', single_energy=None, aug_transforms=False):
        """
        Arguments:
            hdf5_file: path to hdf5 file
            particle_type: photon, pion or electron
            xml_filename: path to XML filename
            transform: list of transformations
        """
        
        self.voxels, self.layer_boundaries = load_data(hdf5_file, particle_type, xml_filename, single_energy=single_energy)
        self.energy, self.layers = get_energy_and_sorted_layers(self.voxels)
        del self.voxels
                
        self.transform = transform
        self.aug_transforms = aug_transforms
        self.device = device
        self.dtype = torch.get_default_dtype()
        
        self.energy = torch.tensor(self.energy, dtype=self.dtype)
        self.layers = torch.tensor(self.layers, dtype=self.dtype)

        # apply preprocessing and then move to GPU
        if self.transform:
            for fn in self.transform:
                self.layers, self.energy = fn(self.layers, self.energy)

        val_size = int(len(self.energy)*val_frac)
        trn_size = len(self.energy) - val_size
        # make train/val split
        if split == 'training':
            self.layers = self.layers[:trn_size]
            self.energy = self.energy[:trn_size]
        elif split == 'validation':
            self.layers = self.layers[-val_size:]
            self.energy = self.energy[-val_size:]
       
        self.layers = self.layers.to(device)
        self.energy = self.energy.to(device)

        print("Dataset loaded, shape: ", self.layers.shape, self.energy.shape)
        print("Device: ", self.energy.device)
       

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, idx):
        if self.aug_transforms:
            randint = torch.randint(high=self.layers.shape[-2], size=(1,))
            indxs = torch.arange(self.layers.shape[-2])
            rot_lay = self.layers[idx, :, :, (indxs+randint)%self.layers.shape[-2]]
            return rot_lay, self.energy[idx], self.layers[idx]
        return self.layers[idx], self.energy[idx]
