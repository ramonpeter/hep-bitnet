import numpy as np
import torch
import torch.nn.functional as F

import Networks
import Models
from Models.ModelBase import GenerativeModel
from Util.util import get
from Util.util import loss_cbvae

from challenge_files import *
from challenge_files import evaluate


class AE(GenerativeModel):

    def __init__(self, params, device, doc):
        super().__init__(params, device, doc)
        self.lat_mean = None
        self.lat_std = None
        self.kl = self.params.get('ae_kl')
        # parameters for autoencoder

    def build_net(self):
        """
        Build the network
        """
        network = get(self.params, "ae_network", "AutoEncoder")
        try:
            return getattr(Networks, network)(self.params).to(self.device)
        except AttributeError:
            raise NotImplementedError(f"build_model: Network class {network} not recognised")
    
    def get_conditions_and_input(self, input):
        """
        :param input: model input + conditional input
        :return: model input, conditional input
        """
        # x = input[0].clone()
        condition = input[1]
        weights = None
        # return x, condition, weights
        return input[0], condition, weights

    def forward(self, x, c):
        
        c = self.net.c_encoding(c)
        z = self.net.encode(x, c)
        if self.params.get('ae_kl', False):
            mu, logvar = z[0], z[1]
            z = self.net.reparameterize(mu, logvar)
            rec = self.net.decode(z, c)
            return rec, mu, logvar
        return self.net.decode(z, c)

    def batch_loss(self, x):
        
        x, c, _ = self.get_conditions_and_input(x)
        x, c = self.flatten_layer_to_batch(x, c)

        #calculate loss for 1 batch
        if self.params.get('ae_kl', False):
            rec, mu, logvar = self.forward(x, c)
        else:
            rec = self.forward(x, c)
        
        loss_fn = self.params.get('ae_loss', 'mse')
        if loss_fn == 'mse':
            loss = torch.mean((x - rec) ** 2)
        elif loss_fn == 'bce':
            loss_fn = torch.nn.BCELoss()
            loss = loss_fn(rec, x)
        elif loss_fn == 'mod-bce':
            rec = rec.reshape(-1, 45, 16*9)
            rec = F.log_softmax(rec, dim=-1)
            rec = rec.reshape(-1, 45, 16, 9)
            loss = -torch.mean(x*rec)
        elif loss_fn == 'bce_mse':
            loss_fn = torch.nn.BCELoss()
            loss_bce = loss_fn(rec, x)
            loss_mse = torch.mean((torch.special.logit(x, eps=1.e-6) - torch.special.logit(rec, eps=1.e-6))**2)
            loss = loss_bce+ 0.0001*loss_mse
        elif loss_fn == 'cbce':
            loss = loss_cbvae(rec, x)
        elif loss_fn == 'bce_reg':
            alpha = self.params.get('bce_reg_alpha', 1.0)
            loss_fn = torch.nn.BCELoss()
            
            loss_reg = torch.mean(enc**2/2)
            loss = loss_fn(rec, x)
            loss += alpha*loss_reg
        elif loss_fn == 'bce_kl':
            loss_fn = torch.nn.BCELoss()
            beta = self.params.get('ae_kl_beta', 1.e-5)
            KLD = -0.5 * torch.mean(1 + logvar - mu**2 -  logvar.exp())
            loss = loss_fn(rec, x) + beta*KLD
        elif loss_fn == 'focal':
            gamma = self.params.get('gamma', 0.5)
            loss_fn = FocalLoss(gamma=gamma)
            loss = loss_fn(rec, x)
        else:
            raise Exception("Unknown loss function")

        return loss

    @torch.inference_mode()
    def sample_batch(self, x):
        x, c, weights = self.get_conditions_and_input(x)
        x, c_flat = self.flatten_layer_to_batch(x, c)
        if self.params.get('ae_kl', False):
            rec, mu, logvar = self.forward(x, c_flat)
        else:
            rec = self.net(x, c_flat)
        rec = self.unflatten_layer_from_batch(rec)
        return rec.detach().cpu(), c.detach().cpu()

    def plot_samples(self, samples, conditions, name="", energy=None, mode='all'): #TODO
        transforms = self.transforms
        print("Plotting reconstructions of input showers")

        for fn in transforms[::-1]:
            samples, conditions = fn(samples, conditions, rev=True)

        samples = samples.detach().cpu().numpy()
        conditions = conditions.detach().cpu().numpy()

        self.save_sample(samples, conditions, name="_ae_reco")
        evaluate.run_from_py(samples, conditions, self.doc, self.params)

    @torch.inference_mode()
    def encode(self, x, c):
        x, c = self.flatten_layer_to_batch(x, c)
        c = self.net.c_encoding(c)
        enc = self.net.encode(x,c)
        return enc

    @torch.inference_mode()
    def decode(self, x, c):
        c = self.net.c_encoding(c)
        x = self.net.decode(x, c)
        return self.unflatten_layer_from_batch(x)

    @torch.inference_mode()
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std*esp
        return z
    
    def flatten_layer_to_batch(self, x, c):
        c = c.repeat_interleave(self.shape[0], 0) # repeat condition for each layer
        x = x.flatten(0,1).unsqueeze(1) # flatten B,L and add channel
        return x, c

    def unflatten_layer_from_batch(self, x):
        x = x.squeeze().unflatten(0, (-1, self.shape[0])) # remove channel and unflatten B,L
        return x

#temp
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(2),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.flatten()

        logpt = F.log_softmax(input, dim=-1).flatten()
        #logpt = logpt.gather(1,target)
        logpt = logpt*target
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


