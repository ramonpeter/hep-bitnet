import math
from typing import Tuple, Iterator

import torch

class MyDataLoader:
    """ A data loader that is way faster than the PyTorch data loader for data with a dimension of less than tausend. """

    data: torch.Tensor
    batch_size: int
    drop_last: bool
    shuffle: bool

    def __init__(self, data: torch.Tensor, cond: torch.Tensor, batch_size: int,
                drop_last:bool=False, shuffle:bool=True, width_noise:float=1e-7, fixed_noise=False) -> None:
        """
            Initializes MyDataLoader class.

            Parameters:
            data (tensor): Data
            cond (tensor): Condition
            batch_size (int): Batch size
            drop_last (bool): If true the last batch is dropped if it is smaller as the given batch size
            shuffle (bool): Weather or not to shuffle the data
            width_noise (float): Width of the noise to add to the data
        """
        self.data = data
        self.cond = cond
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.width_noise  = width_noise
        self.noise_distribution = torch.distributions.Uniform(torch.tensor(0., device=data.device), torch.tensor(1., device=data.device))
        # self.noise_distribution = torch.distributions.Beta(torch.tensor(3., device=data.device), torch.tensor(3., device=data.device))
        
        self.fixed_noise = fixed_noise
        if fixed_noise:
            self.data = self.add_noise(self.data)

        if self.drop_last:
            self.max_batch = len(self.data) // self.batch_size
        else:
            self.max_batch = math.ceil(len(self.data) / self.batch_size)

    def add_noise(self, input: torch.Tensor) -> torch.Tensor:
        noise = self.noise_distribution.sample(input.shape)*self.width_noise
        return input + noise.reshape(input.shape)
    
    def fix_noise(self):
        if self.fixed_noise:
            print("Noise already fixed")
            return 
        else:
            self.fixed_noise = True
            self.data = self.add_noise(self.data)
    
    def drop_last_batch(self):
        self.max_batch = len(self.data) // self.batch_size

    def __len__(self) -> int:
        return self.max_batch

    def __iter__(self) -> Iterator[torch.Tensor]:
        # reset
        if self.shuffle:
            self.index = torch.randperm(len(self.data), device=self.data.device)
        else:
            self.index = torch.arange(len(self.data), device=self.data.device)
        self.batch = 0

        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch >= self.max_batch:
            raise StopIteration
        first = self.batch*self.batch_size
        last = min(first+self.batch_size, len(self.data))
        idx = self.index[first:last]
        self.batch += 1
        if not self.fixed_noise:
            data = torch.clone(self.add_noise(self.data[idx]))
        else:
            data = torch.clone(self.data[idx])
        cond = torch.clone(self.cond[idx])
        return data, cond
