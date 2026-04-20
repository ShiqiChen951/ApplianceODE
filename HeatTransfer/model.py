
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Approximation_block(nn.Module):
    
    def __init__ (self, in_channels, out_channels, modes):
        super(Approximation_block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.float))

    def forward(self, x, LBO_MATRIX, LBO_INVERSE, label = 'True'):
        
        
        ################################################################
        # Encode
        ################################################################
        x = x = x.permute(0, 2, 1)
        x = LBO_INVERSE @ x  
        x = x.permute(0, 2, 1)
            
        ################################################################
        # Approximator
        ################################################################
        x = torch.einsum("bix,iox->box", x[:, :], self.weights1)

        ################################################################
        # Decode
        ################################################################
        x =  x @ LBO_MATRIX.T
        
        return x
    
        
class NORM_net(nn.Module):
    def __init__(self, modes, width, MATRIX_Output, INVERSE_Output, MATRIX_Input, INVERSE_Input):
        super(NORM_net, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width) 
        self.fc01 = nn.Linear(self.width, self.width)
        self.LBO_Matri_input = MATRIX_Input
        self.LBO_Inver_input = INVERSE_Input
        self.LBO_Matri_output = MATRIX_Output
        self.LBO_Inver_output = INVERSE_Output
        
        self.conv_encode = Approximation_block(self.width, self.width, self.modes1)
        
        self.conv1 = Approximation_block(self.width, self.width, self.modes1)
        self.conv2 = Approximation_block(self.width, self.width, self.modes1)
        self.conv3 = Approximation_block(self.width, self.width, self.modes1)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # [10, 186, 2]
        
        x = self.fc0(x)
        x  = F.gelu(x)
        x = self.fc01(x)
        x = x.permute(0, 2, 1) # [10, 64, 186]
        
        x1 = self.conv_encode(x, self.LBO_Matri_input, self.LBO_Inver_input)
        x2 = self.w0(x)
        x  = x1 + x2
        x  = F.gelu(x) # [10,64,186]
        
        '''
        from the Input manifold to the Output manifold
        '''
        x = self.conv1(x, self.LBO_Matri_output, self.LBO_Inver_input)
        x = F.gelu(x) # [10,64,7199]
        
        x1 = self.conv2(x, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w1(x)
        x  = x1  + x2 # [10,64,7199]
        
        x1 = self.conv3(x, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w2(x)
        x  = x1  + x2 # [10,64,7199]
       
        x = x.permute(0, 2, 1)
        
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x) # [10,7199,1]
        
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)     

class NORM_net_rag(nn.Module):
    def __init__(self, modes, width, MATRIX_Output, INVERSE_Output, MATRIX_Input, INVERSE_Input):
        super(NORM_net_rag, self).__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2 + 2, self.width) 
        # self.fc0 = nn.Linear(2 + 1, self.width) 
        self.fc01 = nn.Linear(self.width, self.width)
        self.LBO_Matri_input = MATRIX_Input
        self.LBO_Inver_input = INVERSE_Input
        self.LBO_Matri_output = MATRIX_Output
        self.LBO_Inver_output = INVERSE_Output
        
        self.conv_encode = Approximation_block(self.width, self.width, self.modes1)
        
        self.conv1 = Approximation_block(self.width, self.width, self.modes1)
        self.conv2 = Approximation_block(self.width, self.width, self.modes1)
        self.conv3 = Approximation_block(self.width, self.width, self.modes1)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

        # Ref_score
        ref_score = ref_score.reshape( -1, 1, 1 ) * torch.ones_like( x ) # B, N, 1

        grid = self.get_grid(x.shape, x.device) # B, N, 1
        
        x = torch.cat((x, ref_score, ref_x , grid), dim=-1) # B, N, 4
        # x = torch.cat((x, ref_x , grid), dim=-1) # B, N, 3
        
        x = self.fc0(x)
        x  = F.gelu(x)
        x = self.fc01(x)
        x = x.permute(0, 2, 1)
        
        x1 = self.conv_encode(x, self.LBO_Matri_input, self.LBO_Inver_input)
        x2 = self.w0(x)
        x  = x1 + x2
        x  = F.gelu(x)

        '''
        from the Input manifold to the Output manifold
        '''
        x = self.conv1(x, self.LBO_Matri_output, self.LBO_Inver_input)
        x = F.gelu(x)
        
        x1 = self.conv2(x, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w1(x)
        x  = x1  + x2
        
        x1 = self.conv3(x, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w2(x)
        x  = x1  + x2

        x = x.permute(0, 2, 1)
        
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x + ref_y.reshape(x.shape)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)     

class NORM_Net_DeltaPhi_ODE(nn.Module):
    def __init__(self, modes, width, MATRIX_Output, INVERSE_Output, MATRIX_Input, INVERSE_Input, steps=4):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.steps = steps

        self.fc0 = nn.Linear(2 + 2, width)
        self.fc01 = nn.Linear(width, width)
        self.LBO_Matri_input = MATRIX_Input
        self.LBO_Inver_input = INVERSE_Input
        self.LBO_Matri_output = MATRIX_Output
        self.LBO_Inver_output = INVERSE_Output
        
        self.conv_encode = Approximation_block(width, width, modes)
        self.conv1 = Approximation_block(width, width, modes)
        self.conv2 = Approximation_block(width, width, modes)
        self.conv3 = Approximation_block(width, width, modes)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.padding = 2 # pad the domain if input is non-periodic

        # self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

        ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(x)

        grid = self.get_grid(x.shape, x.device)

        x = torch.cat((x, ref_score, ref_x, grid), dim=-1)  # [10, 186, 4]

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv_encode(x, self.LBO_Matri_input, self.LBO_Inver_input)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        x = self.conv1(x, self.LBO_Matri_output, self.LBO_Inver_input)
        x = F.gelu(x)

        h=x

        # tau = F.softplus(self.tau)
        # dt = tau / self.steps
        depth_scale = 1.0
        for _ in range(self.steps):
            k1 = self.residual_func(h) # [10,64,7199]
            k2 = self.residual_func(h + 0.5 * depth_scale * k1)
            k3 = self.residual_func(h + 0.5 * depth_scale * k2)
            k4 = self.residual_func(h + depth_scale * k3)
            h = h + (depth_scale / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x = h
        # ==================================== #

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x + ref_y.reshape(x.shape)#[10,7199,1]
    def residual_func(self, h):
        x1 = self.conv2(h, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w1(h)
        h_mid = F.gelu(x1 + x2)

        x1 = self.conv3(h_mid, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w2(h_mid)
        dhdt = F.gelu(x1 + x2)
        return dhdt

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx

class NORM_Net_DeltaPhi_ODE2(nn.Module):
    def __init__(self, modes, width, MATRIX_Output, INVERSE_Output, MATRIX_Input, INVERSE_Input, steps=5, coord_dim=2):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.steps = steps

        self.fc0 = nn.Linear(3, width)
        self.fc01 = nn.Linear(width, width)
        self.LBO_Matri_input = MATRIX_Input
        self.LBO_Inver_input = INVERSE_Input
        self.LBO_Matri_output = MATRIX_Output
        self.LBO_Inver_output = INVERSE_Output
        
        self.conv_encode = Approximation_block(width, width, modes)
        self.conv1 = Approximation_block(width, width, modes)
        self.conv2 = Approximation_block(width, width, modes)
        self.conv3 = Approximation_block(width, width, modes)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)

        self.dhdt_expand = nn.Conv1d(self.width, 2*self.width, 1)

        self.coord_proj = nn.Linear(coord_dim, self.width)

        self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.fc3 = nn.Linear(4, self.width)
        self.coord_proj = nn.Linear(coord_dim, self.width)

        self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
        self.padding = 2 # pad the domain if input is non-periodic
        self.coord_dim = coord_dim
        # self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x_coord, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

        ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(ref_y)
        B, N = x['x'].shape[0], x['ref_y'].shape[1]
        ref_score = x['ref_score'].view(B, 1, 1).repeat(1, N, 1)

        ref_y = x['ref_y']
        if ref_y.dim() == 2:
            ref_y = ref_y.unsqueeze(-1)

        grid = self.get_grid((B, N, 1), x['x'].device)
        # print("ref_score:",ref_score.shape,"ref_y:",ref_y.shape,"grid:",grid.shape)
        # ref_score: torch.Size([10, 7199, 1]) ref_y: torch.Size([10, 7199, 1]) grid: torch.Size([10, 7199, 1])
        x_in = torch.cat([ref_score, ref_y, grid], dim=-1)   # (B, N, 2)

        x_feat = self.fc0(x_in)   # (B, N, width)

        a_input = torch.cat([ref_x, x_coord], dim=-1)   # (B, N, 2)
        a_feat = self.fc3(a_input)                     # (B, N, width)

        depth_coord = (x_coord - ref_x) / float(self.steps)   # (B, N, coord_dim)
        depth_feat = self.coord_proj(depth_coord)             # (B, N, width)
        # depth_scale = depth_feat.permute(0, 2, 1).contiguous()  # (B, width, N)
        
        h = x_feat.permute(0, 2, 1).contiguous()    # (B, width, N)
        # a = a_feat.permute(0, 2, 1).contiguous()    # (B, width, N)

        target_N = ref_y.shape[1]

        if a_feat.shape[1] != target_N:
            a = F.interpolate(a_feat.permute(0,2,1), size=target_N, mode='linear', align_corners=False).contiguous()
        if depth_feat.shape[1] != target_N:
            depth_scale = F.interpolate(depth_feat.permute(0,2,1), size=target_N, mode='linear', align_corners=False).contiguous()
  
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
    def func(self, a, h):
        # print("ha1:",ha.shape)
        ha = torch.cat([h, a], dim=1)   # (B, 2*width, N)
        # print("ha2:",ha.shape)
        # ha2: torch.Size([10, 128, 7199])
        ha = self.ha_conv(ha)           # (B, width, N)
        # print("ha1:",ha.shape)
        # ha1: torch.Size([10, 64, 7199])
        h = F.gelu(ha)
        # exit()
        x1 = self.conv_encode(h, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w0(h)
        h = F.gelu(x1 + x2)

        x1 = self.conv1(h, self.LBO_Matri_output, self.LBO_Inver_output)
        h = F.gelu(x1 + x2)

        x1 = self.conv2(h, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w1(h)
        h_mid = F.gelu(x1 + x2)

        x1 = self.conv3(h_mid, self.LBO_Matri_output, self.LBO_Inver_output)
        x2 = self.w2(h_mid)
        dhdt = F.gelu(x1 + x2)

        dhdt = self.dhdt_expand(dhdt)  # (B, 2*width, N)
        dhdt_h, dhdt_a = torch.split(dhdt, self.width, dim=1)

        return dhdt_h

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1).repeat(batchsize, 1, 1)
        return gridx
