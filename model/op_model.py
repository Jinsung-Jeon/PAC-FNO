import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
import math
import numpy as np

class SpectralConv2d_lfno(nn.Module):
    def __init__(self, in_channels, out_channels, mode):
        super(SpectralConv2d_lfno, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = mode-1
        self.modes2 = mode
        self.scale = 1 / in_channels
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def compl_mul2d(self, a, b):
        op = partial(torch.einsum, "bctq,cdtq->bdtq")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x, out_dim):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        x_ft = torch.view_as_real(x_ft)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, 2*self.modes1, self.modes2, 2, device=x.device)
        modes1 = self.modes1 if self.modes1<x_ft.size(-3)//2 else x_ft.size(-3)//2
        modes2 = self.modes2 if self.modes2<x_ft.size(-2) else x_ft.size(-2)
        out_ft[:, :, :modes1, :modes2] = \
            self.compl_mul2d(x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2])
        out_ft[:, :, -modes1:, :modes2] = \
            self.compl_mul2d(x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, -modes1:, :modes2])
        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfft2(out_ft, s=(out_dim,out_dim))
        return x

class lfno_block(nn.Module):
    def __init__(self, in_codim, out_codim, mode):
        super(lfno_block, self).__init__()
        self.w = nn.Conv2d(in_codim, out_codim, 1)
        self.conv = SpectralConv2d_lfno(in_codim, out_codim, mode)
        self.bn = nn.BatchNorm2d(out_codim)
        
    def forward(self, input):
        x, out_dim = input
        out1 = self.conv(x, out_dim)
        out2 = self.w(x)
        out2 = torch.nn.functional.interpolate(out2, size=(out_dim, out_dim),mode='bicubic', align_corners=True)
        out = out1+out2
        out = self.bn(out)
        out = F.relu(out)
        return out, out_dim



class lfno_layer(nn.Module):
    def __init__(self, mode):
        super(lfno_layer, self).__init__()
        self.inplanes = 3
        self.hidden = 3
        self.out = 3
        self.mode = mode
        self.num_blocks = 2
        modules = []
        modules.append(lfno_block(self.inplanes, self.hidden, self.mode))
        for _ in range(self.num_blocks-2):
            modules.append(lfno_block(self.hidden, self.hidden, self.mode))
        modules.append(lfno_block(self.hidden, self.out, self.mode))
        self.layer0 = nn.Sequential(*modules)
        modules = []
        modules.append(lfno_block(self.inplanes, self.hidden, self.mode))
        for _ in range(self.num_blocks-2):
            modules.append(lfno_block(self.hidden, self.hidden, self.mode))
        modules.append(lfno_block(self.hidden, self.out, self.mode))
        self.layer1 = nn.Sequential(*modules)
        self.fc = nn.Linear(self.out*2, self.out)
        self.bn = nn.BatchNorm2d(self.out)
    
    def forward(self, x, out_dim):
        input = x, out_dim
        x1, _ = self.layer0(input)
        x2, _ = self.layer1(input)
        out = torch.cat((x1, x2), dim=1)
        out = out.permute(0,2,3,1).contiguous()
        out = self.fc(out)
        out = out.permute(0,3,1,2).contiguous()
        out = self.bn(out)
        return out



class lfno_layer_large(nn.Module):
    def __init__(self, mode):
        super(lfno_layer_large, self).__init__()
        self.inplanes = 3
        self.hidden = 3
        self.out = 3
        self.mode = mode
        self.num_blocks = 2
        modules = []
        modules.append(lfno_block(self.inplanes, self.hidden, self.mode))
        for _ in range(self.num_blocks-2):
            modules.append(lfno_block(self.hidden, self.hidden, self.mode))
        modules.append(lfno_block(self.hidden, self.out, self.mode))
        self.layer0 = nn.Sequential(*modules)
        modules = []
        modules.append(lfno_block(self.inplanes, self.hidden, self.mode))
        for _ in range(self.num_blocks-2):
            modules.append(lfno_block(self.hidden, self.hidden, self.mode))
        modules.append(lfno_block(self.hidden, self.out, self.mode))
        self.layer1 = nn.Sequential(*modules)
        modules = []
        modules.append(lfno_block(self.inplanes, self.hidden, self.mode))
        for _ in range(self.num_blocks-2):
            modules.append(lfno_block(self.hidden, self.hidden, self.mode))
        modules.append(lfno_block(self.hidden, self.out, self.mode))
        self.layer2 = nn.Sequential(*modules)
        modules = []
        modules.append(lfno_block(self.inplanes, self.hidden, self.mode))
        for _ in range(self.num_blocks-2):
            modules.append(lfno_block(self.hidden, self.hidden, self.mode))
        modules.append(lfno_block(self.hidden, self.out, self.mode))
        self.layer3 = nn.Sequential(*modules)
        self.fc = nn.Linear(self.out*4, self.out)
        self.bn = nn.BatchNorm2d(self.out)
    
    def forward(self, x, out_dim):
        input = x, out_dim
        x1, _ = self.layer0(input)
        x2, _ = self.layer1(input)
        x3, _ = self.layer2(input)
        x4, _ = self.layer3(input)
        out = torch.cat((x1, x2, x3, x4), dim=1)
        out = out.permute(0,2,3,1).contiguous()
        out = self.fc(out)
        out = out.permute(0,3,1,2).contiguous()
        out = self.bn(out)
        return out



class lfno_layer_small(nn.Module):
    def __init__(self, mode):
        super(lfno_layer_small, self).__init__()
        self.inplanes = 3
        self.hidden = 3
        self.out = 3
        self.mode = mode
        modules = []
        modules.append(lfno_block(self.inplanes, self.out, self.mode))
        self.layer0 = nn.Sequential(*modules)
        modules = []
        modules.append(lfno_block(self.inplanes, self.out, self.mode))
        self.layer1 = nn.Sequential(*modules)
        self.fc = nn.Linear(self.out*2, self.out)
        self.bn = nn.BatchNorm2d(self.out)
    
    def forward(self, x, out_dim):
        input = x, out_dim
        x1, _ = self.layer0(input)
        x2, _ = self.layer1(input)
        out = torch.cat((x1, x2), dim=1)
        out = out.permute(0,2,3,1).contiguous()
        out = self.fc(out)
        out = out.permute(0,3,1,2).contiguous()
        out = self.bn(out)
        return out





class SpectralConv2d_fno(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fno, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    # Complex multiplication
    def compl_mul2d(self, a, b):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        op = partial(torch.einsum, "bixy,ioxy->boxy")
        return torch.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x, out_dim):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        x_ft = torch.view_as_real(x_ft)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, 2*self.modes1, self.modes2, 2, device=x.device)
        
        modes1 = self.modes1 if self.modes1<x_ft.size(-3)//2 else x_ft.size(-3)//2
        modes2 = self.modes2 if self.modes2<x_ft.size(-2) else x_ft.size(-2)
        out_ft[:, :, :modes1, :modes2] = \
            self.compl_mul2d(x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2])
        out_ft[:, :, -modes1:, :modes2] = \
            self.compl_mul2d(x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, -modes1:, :modes2])

        #Return to physical space
        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfft2(out_ft, s=(out_dim,out_dim))
        return x

class FNO2d(nn.Module):
    def __init__(self, mode):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = 12
        self.modes2 = 12
        self.width = 32
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(5, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d_fno(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fno(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fno(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fno(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x, out_dim):
        out_dim = out_dim+self.padding
        x = x.permute(0, 2, 3, 1).contiguous()
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x, out_dim)
        x2 = self.w0(x)
        x2 = torch.nn.functional.interpolate(x2, size=(out_dim, out_dim),mode='bicubic', align_corners=True)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x, out_dim)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x, out_dim)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x, out_dim)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.bn(x)
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)