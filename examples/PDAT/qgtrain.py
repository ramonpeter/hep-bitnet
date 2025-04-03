import os
import numpy as np
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from torch.autograd import Variable
import multiprocessing
from multiprocessing import Process
#torch.manual_seed(6)

device = torch.device("cuda")
    
class BitLinear(nn.Linear):
    """
    BitLinear is a custom linear layer that performs quantization of weights and activations

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        b (int, optional): Number of bits for quantizatio. Defaults to 8.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias)
        self.eps = 1e-8
        self.device = self.weight.device
        self.dtype = self.weight.dtype

        # Quantiziation and dequantization
        self.Q_b = 2 ** (b - 1) - 1.0
        self.beta = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.gamma = torch.tensor(0.0, device=self.device, dtype=self.dtype)

    def quantize_weights(self, w: Tensor) -> Tensor:
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        alpha = w.mean()
        self.beta = w.abs().mean().clamp_(min=self.eps)
        quantized_weight = torch.sign(w - alpha)

        return quantized_weight * self.beta

    def quantize_activations(self, x: Tensor) -> Tensor:
        """
        Quantizes the activations of the layer.

        Args:
            x (Tensor): Input tensor.
            b (int, optional): Number of bits for quantization. Default is 8.

        Returns:
            Tensor: Quantized activations tensor.
        """
        self.gamma = self.Q_b / x.abs().max(dim=-1, keepdim=True).values.clamp_(
            min=self.eps
        )
        quantized_x = (x * self.gamma).round().clamp_(-(self.Q_b + 1), self.Q_b)

        return quantized_x / self.gamma

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # weight tensor with shape (in_features, out_features)
        w = self.weight

        # Quantize weights
        w_quant = w + (self.quantize_weights(w) - w).detach()

        # Quantize input
        x_quant = x + (self.quantize_activations(x) - x).detach()

        # Perform linear transformation
        output = F.linear(x_quant, w_quant, self.bias)

        # Return dequantized output
        return output


class BitLinear158b(BitLinear):
    """
    BitLinear158b layer allowing for tertiar weights (-1,0,1). Rest is keeped
    as in BitLinear

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        b (int, optional): Number of bits for quantizatio. Defaults to 8.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        b: int = 8,
    ):
        super().__init__(in_features, out_features, bias, b)

    def quantize_weights(self, w: Tensor):
        """
        Quantizes the weights using the absmean quantization function.

        Returns:
            Tensor: Quantized weight tensor.
        """
        self.beta = w.abs().mean().clamp_(min=self.eps)
        quantized_weight = (w / self.beta).round().clamp_(-1, 1)

        return quantized_weight * self.beta
            
class PointCloudDataset(data.Dataset):
    def __init__(self, data, labels, interact1, interact2):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.interact1 = torch.tensor(interact1, dtype=torch.float32)
        self.interact2 = torch.tensor(interact2, dtype=torch.float32)
    def __getitem__(self, index):
        return self.data[index], self.labels[index], self.interact1[index], self.interact2[index]

    def __len__(self):
        return len(self.data)

# 定义边缘卷积函数
class EdgeConv1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, num_points, num_neighbors, num_features)
        x = self.conv1(x.permute(0, 3, 1, 2))  # (batch_size, out_channels, num_points, 11)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)
        x = torch.mean(x, dim=3)  # 对每个点周围的21个邻居点取平均值
        return x


# 定义边缘卷积函数
class EdgeConv2(nn.Module):
    def __init__(self, in_channels, out_channels, num_neighbors):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        # x: (batch_size, num_points, num_features)
        batch_size, num_points, num_features = x.size()

        # Compute pairwise distances and find nearest neighbors
        pairwise_distances = torch.sum((x.unsqueeze(2) - x.unsqueeze(1))**2, dim=-1)  # (batch_size, num_points, num_points)
        _, indices = torch.topk(pairwise_distances, k=self.num_neighbors, dim=-1, largest=False)  # Nearest neighbor indices

        neighbor_features = x.unsqueeze(2).expand(-1, -1, num_points, -1)  # (batch_size, num_points, num_points, num_features)
        neighbor_features = torch.gather(neighbor_features, 2, indices.unsqueeze(-1).expand(-1, -1, -1, num_features))

        x = self.conv1(neighbor_features.permute(0, 3, 1, 2))  # (batch_size, out_channels, num_points, num_neighbors)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.gelu(x)
        x = torch.mean(x, dim=3)  # Average pooling over neighbors
        return x
                
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (self.head_dim * num_heads == embed_dim), "Embedding dimension must be divisible by number of heads"

        self.query = BitLinear158b(embed_dim, embed_dim)
        self.key = BitLinear158b(embed_dim, embed_dim)
        self.value = BitLinear158b(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.fcx = BitLinear158b(embed_dim, embed_dim)
        self.fc1 = BitLinear158b(embed_dim, embed_dim)
        self.fc2 = BitLinear158b(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv2d(6,32,kernel_size=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(p = 0.1)
        self.conv2 = nn.Conv2d(32,16,kernel_size=1)
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.dropout2 = nn.Dropout(p = 0.1)
        self.conv3 = nn.Conv2d(16,8,kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.dropout3 = nn.Dropout(p = 0.1)
        self.gelu = nn.GELU()
        self.dropout5 = nn.Dropout(p = 0.1)
        self.dropout6 = nn.Dropout(p = 0.1)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ln4 = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, x, interact1):
        residual1 = x
        batch_size, seq_len, embed_dim = x.size()
        x = self.ln1(x)
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        interact1 = self.conv1(interact1.permute(0,3,1,2))
        interact1 = self.bn1(interact1)
        interact1 = self.gelu(interact1)
        interact1 = self.dropout1(interact1)
        interact1 = self.conv2(interact1)
        interact1 = self.bn2(interact1)
        interact1 = self.gelu(interact1)
        interact1 = self.dropout2(interact1)
        interact1 = self.conv3(interact1)
        interact1 = self.bn3(interact1)
        interact1 = self.gelu(interact1)
        interact1 = self.dropout3(interact1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)) 
        scores += interact1         
        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        attn_output = attn_output - x
        batch_size, num_points, num_features = attn_output.size()
        attn_output = attn_output.view(batch_size * num_points, num_features)
        attn_output = self.gelu(self.fcx(attn_output))
        attn_output = attn_output.view(batch_size, num_points, -1)
        attn_output = self.dropout(attn_output)
        attn_output = self.ln2(attn_output)
        residual2 = attn_output + residual1
        attn_output = self.ln3(residual1 + attn_output)
        attn_output = self.fc1(attn_output)
        attn_output = self.gelu(attn_output)
        attn_output = self.dropout5(attn_output)
        attn_output = self.ln4(attn_output)
        attn_output = self.fc2(attn_output)
        attn_output = self.dropout6(attn_output)
        attn_output += residual2
        return attn_output
    

class ChannelAttention(nn.Module):
    def __init__(self, embed_dim, interact_dim, dropout = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = BitLinear158b(embed_dim, embed_dim)
        self.key = BitLinear158b(embed_dim, embed_dim)
        self.value = BitLinear158b(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = BitLinear158b(embed_dim, embed_dim)
        self.fc2 = BitLinear158b(embed_dim, embed_dim)
        self.fcx = BitLinear158b(interact_dim**2, embed_dim**2)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.interact_dim = interact_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)
        self.ln4 = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(p = 0.1)
        self.dropout2 = nn.Dropout(p = 0.1)
        self.dropout3 = nn.Dropout(p = 0.1)
        self.dropout = nn.Dropout(dropout)
        self.upsample = nn.Upsample(size=(64,64), mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(1,1,kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(1)
        self.dropout4 = nn.Dropout(p = 0.1)
        self.conv2 = nn.Conv2d(1,1,kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1)
        self.dropout5 = nn.Dropout(p = 0.1)
        self.conv3 = nn.Conv2d(1,1,kernel_size=3, stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1)
        self.dropout6 = nn.Dropout(p = 0.1)

        
    def forward(self, x, interact2):
        residual1 = x
        batch_size, seq_len, embed_dim = x.size()
        x = self.ln1(x)
        q = self.query(x).view(batch_size, embed_dim, seq_len)
        k = self.key(x).view(batch_size, embed_dim, seq_len)
        v = self.value(x).view(batch_size, embed_dim, seq_len)
        interact2 = interact2.unsqueeze(dim=-1)
        interact2 = self.upsample(interact2.permute(0,3,1,2))
        interact2 = self.conv1(interact2)
        interact2 = self.gelu(interact2)
        interact2 = self.dropout4(interact2)
        interact2 = self.conv2(interact2)
        interact2 = self.gelu(interact2)
        interact2 = self.dropout5(interact2)
        interact2 = self.conv3(interact2)
        interact2 = self.gelu(interact2)
        interact2 = self.dropout6(interact2)
        interact2 = interact2.squeeze(dim=1)
        scores = torch.bmm(q,k.transpose(-2, -1))/torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        scores += interact2
        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  
        attn_output = attn_output - x
        attn_output = self.dropout(attn_output) 
        attn_output = self.ln2(attn_output)
        residual2 = attn_output + residual1
        attn_output = self.ln3(attn_output + residual1)
        attn_output = self.fc1(attn_output)
        attn_output = self.gelu(attn_output)
        attn_output = self.dropout2(attn_output)
        attn_output = self.ln4(attn_output)
        attn_output = self.fc2(attn_output)
        attn_output = self.dropout3(attn_output)
        attn_output += residual2
        return attn_output

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 2)
        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.softmax(self.fc2(x))
        return x     
    
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = EdgeConv1(8, 128)
        self.conv2 = EdgeConv2(128, 64, 20)
        self.att1 = SelfAttention(64, 8)
        self.att2 = ChannelAttention(64, 8)
        self.mlp = MLP()
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(256, 64)
        self.bn = nn.BatchNorm1d(64)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.5)
        self.conv0 = nn.Conv1d(320, 256, kernel_size=1, stride = 1)        
    def forward(self, x, interact1, interact2):
        x = self.conv1(x)
        x = x.permute(0,2,1)
        x = self.conv2(x)
        x = x.permute(0,2,1)
        x1 = self.att1(x, interact1)
        x2 = self.att2(x1, interact2)
        x3 = self.att1(x2, interact1)
        x4 = self.att2(x3, interact2)
        concatenated_output = torch.cat([x,x1,x2,x3,x4],dim = -1)
        concatenated_output = concatenated_output.permute(0,2,1)
        concatenated_output = self.conv0(concatenated_output)
        concatenated_output = concatenated_output.permute(0,2,1)
        x = torch.mean(concatenated_output, dim=1)
        output_x = x.squeeze()
        final_output = self.mlp(output_x)
        return final_output

model = MyModel().to(device)
state_dict = torch.load('/users/daohan.wang/DAT/qgc/2/weights/model-010.weights')
model.load_state_dict(state_dict)


optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 读取数据文件
train_files = []
for i in range(1, 1281):
    file_path = f"/groups/hephy/mlearning/daohan/qg/new/new/train_data_with_neighbors{i}.npz"
    train_files.append(file_path)

valid_files = []
for i in range(1, 161):
    file_path = f"/groups/hephy/mlearning/daohan/qg/new/new/valid_data_with_neighbors{i}.npz"
    valid_files.append(file_path)

for epoch in range(100):
    print('\033[1;32m', 'Epoch', epoch, '\033[0m')

    train_losses = []
    valid_losses = []
    train_loss = 0.0
    valid_loss = 0.0
    m = 0
    n = 0

    for i in range(1280):
     #   print('batch_number:'+str(i+1))
        train_data = np.load(train_files[i], mmap_mode='r')['data']
        train_labels = np.load(train_files[i], mmap_mode='r')['labels']
        train_interact1 = np.load('/groups/hephy/mlearning/daohan/qg/new/new/train_interact' + str(i+1) + '.npz', mmap_mode='r')['interact1']
        train_interact2 = np.load('/groups/hephy/mlearning/daohan/qg/new/new/train_interact_' + str(i+1) + '.npy', mmap_mode='r')
        train_dataset = PointCloudDataset(train_data, train_labels, train_interact1, train_interact2)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

        model.train()
        pbar = tqdm(train_loader, total=len(train_loader))
        for PC, labels, interact1, interact2 in pbar:
            PC = PC.to(device)
            labels = labels.to(device)
            interact1 = interact1.to(device)
            interact2 = interact2.to(device)
            output = model(PC, interact1, interact2)
            loss = F.cross_entropy(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = (train_loss * m + loss.item()) / (m + 1)
            m += 1
            pbar.set_description('train loss %.7f' % train_loss)

            del PC
            del labels
            del interact1
            del interact2
            torch.cuda.empty_cache()
        del train_data
        del train_labels
        del train_interact1
        del train_interact2
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), '/users/daohan.wang/DAT/qgc/3/weights/model-%.3d.weights' % epoch)
    scheduler.step()


    for j in range(160):
        valid_data = np.load(valid_files[j], mmap_mode='r')['data']
        valid_labels = np.load(valid_files[j], mmap_mode='r')['labels']
        valid_interact1 = np.load('/groups/hephy/mlearning/daohan/qg/new/new/valid_interact' + str(j+1) + '.npz', mmap_mode='r')['interact1']
        valid_interact2 = np.load('/groups/hephy/mlearning/daohan/qg/new/new/valid_interact_' + str(j+1) + '.npy', mmap_mode='r')
        valid_dataset = PointCloudDataset(valid_data, valid_labels, valid_interact1, valid_interact2)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=True)
      
        model.eval()
        with torch.no_grad():
            pbar = tqdm(valid_loader, total=len(valid_loader))
            for PC, labels, interact1, interact2 in pbar:
                PC = PC.to(device)
                labels = labels.to(device)
                interact1 = interact1.to(device)
                interact2 = interact2.to(device)
                output = model(PC, interact1, interact2)
                loss = F.cross_entropy(output, labels)
                valid_loss = (valid_loss * n + loss.item()) / (n + 1)
                n += 1
                pbar.set_description('valid loss %.7f' % valid_loss)
                del PC
                del labels
                del interact1 
                del interact2
                torch.cuda.empty_cache()
        del valid_data
        del valid_labels
        del valid_interact1 
        del valid_interact2
        torch.cuda.empty_cache()
            
    print('train loss', train_loss, 'valid loss', valid_loss)
    with open('/users/daohan.wang/DAT/qgc/3/results.txt', 'a') as f:
        f.write(f'{epoch} {train_loss} {valid_loss}\n')            
            
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
