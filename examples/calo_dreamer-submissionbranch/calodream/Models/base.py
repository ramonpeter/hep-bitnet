import numpy as np
import torch
import torch.nn as nn
import os
import yaml
from torch.utils.data import DataLoader

from bitnet.bitlinear import BitLinear

from calodream import Transforms

class GenerativeModel(nn.Module):

    def __init__(self, params, device, model_dir):
        """
        :param params: file with all relevant model parameters
        """
        super().__init__()
        self.params = params
        self.device = device
        self.shape = self.params['shape']
        self.single_energy = self.params.get( 'single_energy', None)
        self.net = self.build_net()

        self.transforms = get_transformations(params.get('transforms', None), model_dir)
        
        param_count = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print(f'[CaloDREAM] {self.__class__.__name__} has {param_count} parameters')
        # Claudius added
        #print(f"Model: \n")
        #print(self.net)
        # Claudius end
        self.params['parameter_count'] = param_count

    def build_net(self):
        pass

    def generate_Einc_ds1(self, energy=None, sample_multiplier=1000):
        """ generate the incident energy distribution of CaloChallenge ds1
            sample_multiplier controls how many samples are generated: 10* sample_multiplier for low energies,
            and 5, 3, 2, 1 times sample multiplier for the highest energies

        """
        ret = np.logspace(8, 18, 11, base=2)
        ret = np.tile(ret, 10)
        ret = np.array(
            [*ret, *np.tile(2. ** 19, 5), *np.tile(2. ** 20, 3), *np.tile(2. ** 21, 2), *np.tile(2. ** 22, 1)])
        ret = np.tile(ret, sample_multiplier)
        if energy is not None:
            ret = ret[ret == energy]
        np.random.shuffle(ret)
        return ret

    @torch.inference_mode()
    def sample(self, size=10**5, batch_size=10**4):

        self.eval()

        # sample incident energies
        energies = torch.tensor(
            10**np.random.uniform(3, 6, size=size) 
            if self.params['eval_dataset'] in ['2', '3'] else
            self.generate_Einc_ds1(energy=self.single_energy),
            dtype=torch.get_default_dtype(),
            device=self.device
        ).unsqueeze(1)
        
        # transform Einc to basis used in training
        dummy = None
        for fn in self.transforms:
            if hasattr(fn, 'cond_transform'):
                dummy, energies = fn(dummy, energies)

        energies_loader = DataLoader(
            dataset=energies, batch_size=batch_size, shuffle=False
        )
        
        # sample u_i's if self is a shape model
        if self.params['model_type'] == 'shape': 
            
            # load energy model
            energy_model = self.load_other(self.params['energy_model'])
            #raise # shortcut

            # sample us
            u_samples = torch.vstack([
                energy_model.sample_batch(c) for c in energies_loader
            ])
            energies = torch.cat([energies, u_samples], dim=1)
            
            # concatenate with Einc
            energies_loader = DataLoader(
                dataset=energies, batch_size=batch_size, shuffle=False
            )

        # sample batches and stack     
        samples = torch.vstack([self.sample_batch(c) for c in energies_loader])

        # postproces
        showers = samples
        for fn in self.transforms[::-1]:
            showers, energies = fn(showers, energies, rev=True)
        
        return showers.cpu().numpy(), energies.cpu().numpy()

    def sample_batch(self, batch):
        pass

    def load(self, path):
        """ Load the model, and more if needed"""
        state_dicts = torch.load(path, map_location=self.device)
        self.net.load_state_dict(state_dicts["net"])
        # Claudius
        print("Loaded model: \n")
        print(self.net)
        # claudius checking num params:
        total_parameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print("by Claudius: Setup has {} parameters".format(int(total_parameters)))
        bit_parameters = 0
        regular_parameters = 0
        for mod in self.net.modules():
            bit_t, reg_t = count_bit(mod)
            bit_parameters = bit_parameters + bit_t
            regular_parameters = regular_parameters + reg_t
        print(f"by Claudius: Setup has {bit_parameters} bit parameters and {regular_parameters} regular parameters, resulting in total {bit_parameters+regular_parameters} parameters.")
        # end Claudius
        self.net.to(self.device)

    def load_other(self, model_dir):
        """ Load a different model (e.g. to sample u_i's)"""
        
        with open(os.path.join(model_dir, 'params.yaml')) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        model_class = params['model']
        # choose model
        if model_class == 'TBD':
            Model = self.__class__
        if model_class == 'TransfusionAR':
            from calodream import TransfusionAR
            Model = TransfusionAR

        # load model
        other = Model(params, self.device, model_dir)
        state_dicts = torch.load(
            os.path.join(model_dir, 'model.pt'), map_location=self.device
        )
        other.net.load_state_dict(state_dicts["net"])

        # Claudius
        print("Loaded model: \n")
        print(other.net)
        # claudius checking num params:
        total_parameters = sum(p.numel() for p in other.net.parameters() if p.requires_grad)
        print("by Claudius: Setup has {} parameters".format(int(total_parameters)))
        bit_parameters = 0
        regular_parameters = 0
        for mod in other.net.modules():
            bit_t, reg_t = count_bit(mod)
            bit_parameters = bit_parameters + bit_t
            regular_parameters = regular_parameters + reg_t
        print(f"by Claudius: Setup has {bit_parameters} bit parameters and {regular_parameters} regular parameters, resulting in total {bit_parameters+regular_parameters} parameters.")
        # end Claudius

        # use eval mode and freeze weights
        other.eval()
        for p in other.parameters():
            p.requires_grad = False

        return other
    
def get_transformations(transforms_list, model_dir=None):
    func = []
    for name, kwargs in transforms_list.items():
        if name == 'StandardizeFromFile' and model_dir is not None:
            kwargs['model_dir'] = model_dir
        func.append(getattr(Transforms, name)(**kwargs))
    return func

def count_bit(module):
    # count parameters in module, and substract parameters of direct children to only have local number
    # by Claudius
    bit_p = 0
    reg_p = 0
    bit_ch = 0
    reg_ch = 0
    
    if isinstance(module, BitLinear):
        bit_p = bit_p + sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        reg_p = reg_p + sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    for ch in list(module.children()):
        if isinstance(module, BitLinear):
            bit_ch = bit_ch + sum(p.numel() for p in ch.parameters() if p.requires_grad)
        else:
            reg_ch = reg_ch + sum(p.numel() for p in ch.parameters() if p.requires_grad)

    return bit_p - bit_ch, reg_p - reg_ch
