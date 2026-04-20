
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from torchdiffeq import odeint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Approximation_block(nn.Module):
    def __init__ (self, in_channels, out_channels, modes, LBO_MATRIX, LBO_INVERSE):
        
        super(Approximation_block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = LBO_MATRIX.shape[1]
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.float))

    def forward(self, x):
                        
        ################################################################
        # Encode
        ################################################################
        x = x = x.permute(0, 2, 1)
        x = self.LBO_INVERSE @ x  
        x = x.permute(0, 2, 1)
        
        ################################################################
        # Approximator
        ################################################################
        x = torch.einsum("bix,iox->box", x[:, :], self.weights1)
        
        ################################################################
        # Decode
        ################################################################
        x =  x @ self.LBO_MATRIX.T
        
        return x
    
        
class NORM_Net(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE):
        super(NORM_Net, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) 
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.conv0 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv1 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv2 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        self.conv3 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE )
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)     

class NORM_Net_DeltaPhi(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE):
        super(NORM_Net_DeltaPhi, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(2 + 3, self.width) 
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.conv0 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv1 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv2 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv3 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

        ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(x)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, ref_score, ref_x, ref_y.reshape(x.shape), grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)

        return x + ref_y.reshape(x.shape)  

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx

class NORM_Net_ODE(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE, steps=4):
        super(NORM_Net_ODE, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(2 + 3, self.width)  
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.conv0 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv1 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv2 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv3 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.steps = steps

    def func(self, h):
        x1 = self.conv0(h)
        x2 = self.w0(h)
        h = F.gelu(x1 + x2)

        x1 = self.conv1(h)
        x2 = self.w1(h)
        h = F.gelu(x1 + x2)

        x1 = self.conv2(h)
        x2 = self.w2(h)
        h = F.gelu(x1 + x2)

        x1 = self.conv3(h)
        x2 = self.w3(h)
        dhdt = x1 + x2

        return dhdt

    def forward(self, x):
        x, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

        ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(x)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, ref_score, ref_x, ref_y.reshape(x.shape), grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        h = x

        depth_scale=1.0

        for _ in range(self.steps):
            k1 = self.func(h)
            k2 = self.func(h + 0.5 * depth_scale * k1)
            k3 = self.func(h + 0.5 * depth_scale * k2)
            k4 = self.func(h + depth_scale * k3)
            h = h + (depth_scale / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x = h

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)

        return x + ref_y.reshape(x.shape)  

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx 

class NORM_Net_ODE2(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE, steps=5, coord_dim=1):
        super(NORM_Net_ODE2, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(3, self.width)  
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.conv0 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv1 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv2 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv3 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.dhdt_expand = nn.Conv1d(self.width, 2*self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.fc3 = nn.Linear(2, self.width)

        self.coord_proj = nn.Linear(coord_dim, self.width)

        self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
        self.steps = steps
        self.coord_dim = coord_dim

    def func(self, a, h):
        # a, h: (B, width, N)
        ha = torch.cat([h, a], dim=1)   # (B, 2*width, N)
        ha = self.ha_conv(ha)           # (B, width, N)
        h = F.gelu(ha)

        x1 = self.conv0(h)
        x2 = self.w0(h)
        h = F.gelu(x1 + x2)

        x1 = self.conv1(h)
        x2 = self.w1(h)
        h = F.gelu(x1 + x2)

        x1 = self.conv2(h)
        x2 = self.w2(h)
        h = F.gelu(x1 + x2)

        x1 = self.conv3(h)
        x2 = self.w3(h)
        dhdt = x1 + x2                 # (B, width, N)

        dhdt = self.dhdt_expand(dhdt)  # (B, 2*width, N)
        dhdt_h, dhdt_a = torch.split(dhdt, self.width, dim=1)

        return dhdt_h               # (B, width, N)

    def forward(self, x):
        x_coord, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

        ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(ref_y)   # (B, N, 1)
        grid = self.get_grid(ref_y.shape, ref_y.device)                    # (B, N, 1)
        B, N = x['x'].shape[0], x['x'].shape[1]
        ref_score = x['ref_score'].view(B, 1, 1).repeat(1, N, 1)

        ref_y = x['ref_y']
        if ref_y.dim() == 2:
            ref_y = ref_y.unsqueeze(-1)

        grid = self.get_grid((B, N, 1), x['x'].device)

        x_in = torch.cat([ref_score, ref_y, grid], dim=-1)   # (B, N, 3)

        x_feat = self.fc0(x_in)   # (B, N, width)

        a_input = torch.cat([ref_x, x_coord], dim=-1)   # (B, N, 2)
        a_feat = self.fc3(a_input)                     # (B, N, width)

        depth_coord = (x_coord - ref_x) / float(self.steps)   # (B, N, coord_dim)
        depth_feat = self.coord_proj(depth_coord)             # (B, N, width)
        depth_scale = depth_feat.permute(0, 2, 1).contiguous()  # (B, width, N)
        
        h = x_feat.permute(0, 2, 1).contiguous()    # (B, width, N)
        a = a_feat.permute(0, 2, 1).contiguous()    # (B, width, N)

        for _ in range(self.steps):
            k1 = self.func(a, h)
            k2 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k1)
            k3 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k2)
            k4 = self.func(a + depth_scale, h + depth_scale * k3)
            h = h + (depth_scale / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x_out = h.permute(0, 2, 1).contiguous()   # (B, N, width)
        x_out = F.gelu(self.fc1(x_out))            # (B, N, 128)
        x_out = self.fc2(x_out)                    # (B, N, 1)

        return x_out + ref_y.reshape(x_out.shape)  # (B, N, 1)

    def get_grid(self, shape, device):
        # shape is expected to be like (B, N, 1) or (B,N,...); use shape[:2]
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx
    

class NORM_Net_ODE3(nn.Module):
    def __init__(self, modes, width, LBO_MATRIX, LBO_INVERSE, steps=3, coord_dim=1):
        super(NORM_Net_ODE3, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(2 + 3, self.width)  
        self.LBO_MATRIX = LBO_MATRIX
        self.LBO_INVERSE = LBO_INVERSE

        self.conv0 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv1 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv2 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        self.conv3 = Approximation_block(self.width, self.width, self.modes1, self.LBO_MATRIX, self.LBO_INVERSE)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        # self.dhdt_expand = nn.Conv1d(self.width, 2*self.width, 1)

        self.fc1 = nn.Linear(self.width, 256)
        self.fc2 = nn.Linear(256, 1)

        # self.fc3 = nn.Linear(2, self.width)

        # self.coord_proj = nn.Linear(coord_dim, self.width)

        # self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
        self.steps = steps
        self.coord_dim = coord_dim
        

    def func(self, x, ref_score, ref_x, grid, h):
        # a, h: (B, width, N)
        x = torch.cat([x, ref_score, ref_x, grid, h.reshape(x.shape)], dim=-1)   # (B, 2*width, N)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        x, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']
        ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(x)
        grid = self.get_grid(x.shape, x.device)

        depth_coord = (x - ref_x) / float(self.steps)   # (B, N, coord_dim)
    
        h = ref_y.reshape(x.shape)
        int_x = ref_x
        for _ in range(self.steps):
            # sim_score = F.cosine_similarity(x, int_x, dim=1)[:, 0].reshape(-1, 1, 1) * torch.ones_like(x)
            # k1 = self.func(x, sim_score, int_x, grid, h)
            
            # sim_score = F.cosine_similarity(x, int_x + 0.5 * depth_coord, dim=1)[:, 0].reshape(-1, 1, 1) * torch.ones_like(x)
            # k2 = self.func(x, sim_score, int_x + 0.5 * depth_coord, grid, h + 0.5 / float(self.steps) * k1)
            # k3 = self.func(x, sim_score, int_x + 0.5 * depth_coord, grid, h + 0.5 / float(self.steps) * k2)
            
            # sim_score = F.cosine_similarity(x, int_x + depth_coord, dim=1)[:, 0].reshape(-1, 1, 1) * torch.ones_like(x)
            # k4 = self.func(x, sim_score, int_x + depth_coord, grid, h + 1 / float(self.steps) * k3)
            
            # h = h + (1/ float(self.steps) / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) 
            # int_x = int_x + depth_coord
            
            
            sim_score = F.cosine_similarity(x, int_x, dim=1)[:, 0].reshape(-1, 1, 1) * torch.ones_like(x)
            k1 = self.func(x, sim_score, int_x, grid, h)
            h = h + 1/ float(self.steps) * k1
            int_x = int_x + depth_coord

        return h

    def get_grid(self, shape, device):
        # shape is expected to be like (B, N, 1) or (B,N,...); use shape[:2]
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx
    
    
# class NORM_Net_DeltaPhi(nn.Module):
#     def forward(self, x):
#         x, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

#         ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(x)
#         grid = self.get_grid(x.shape, x.device)
#         x = torch.cat((x, ref_score, ref_x, ref_y.reshape(x.shape), grid), dim=-1)

#         x = self.fc0(x)
#         x = x.permute(0, 2, 1)

#         x1 = self.conv0(x)
#         x2 = self.w0(x)
#         x = F.gelu(x1 + x2)

#         x1 = self.conv1(x)
#         x2 = self.w1(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv2(x)
#         x2 = self.w2(x)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv3(x)
#         x2 = self.w3(x)
#         x = x1 + x2

#         x = x.permute(0, 2, 1)
#         x = F.gelu(self.fc1(x))
#         x = self.fc2(x)

#         return x + ref_y.reshape(x.shape)  

#     def get_grid(self, shape, device):
#         batchsize, size_x = shape[0], shape[1]
#         gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
#         gridx = gridx.reshape(1, size_x, 1).repeat(batchsize, 1, 1)
#         return gridx
