
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Spatial_Approximation_block(nn.Module):
    def __init__ (self, in_channels, out_channels, modes, Fmodes, LBO_MATRIX, LBO_INVERSE, TIME_MATRIX, TIME_INVERSE):
        super(Spatial_Approximation_block, self).__init__()
        
        '''
        Approximation_block for space-dimension
        '''
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.modes1 = modes 
        self.modes2 = Fmodes 
        self.LBO_MATRIX = LBO_MATRIX 
        self.LBO_INVERSE = LBO_INVERSE 
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.modes1, in_channels, out_channels, dtype=torch.float)) 
        
                      
    def forward(self, x):  
        
        x = x.permute(0,3,1,2) 
        x = torch.einsum("txbi,xio->txbo", x, self.weights1) 
        return x
    
  
    
class Temporal_Approximation_block(nn.Module):
    def __init__ (self, in_channels, out_channels, modes, Fmodes, LBO_MATRIX, LBO_INVERSE, TIME_MATRIX, TIME_INVERSE):
        super(Temporal_Approximation_block, self).__init__()
        
        '''
        Approximation_block for time-dimension
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels 
        self.modes1 = modes              
        self.modes2 = Fmodes 
        self.TIME_MATRIX = TIME_MATRIX
        self.TIME_INVERSE = TIME_INVERSE 

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.modes2, in_channels, out_channels, dtype=torch.cfloat)) 
        
    def compl_mul1d(self, input, weights):
        
        return torch.einsum("xtbi,tio->xtbo", input, weights)
                      
    def forward(self, x): 
    
        out = torch.zeros(self.modes1, x.size(1), x.size(2), self.in_channels, device=x.device, dtype=torch.float)  
        out[:, :self.modes2, :, :] = self.compl_mul1d(x[:, :self.modes2, :, :], self.weights1) 
        
        return out
    
class Spatiotemporal_Parameterization(nn.Module):
    def __init__ (self, nodes1, nodes2, width):
        super(Spatiotemporal_Parameterization, self).__init__()

        '''
        Approximation_block for space&time-dimension
        '''
        self.in_channels = nodes1 
        self.out_channels = nodes2
        self.modes1 =  width 
        
        self.scale = (1 / (self.in_channels*self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.modes1, self.in_channels, self.out_channels, dtype=torch.float)) 
        
    def forward(self, x):  
        
        x = x.permute(0,3,1,2) 
        x = torch.einsum("txbi,xio->txbo", x, self.weights1) 
        
        return x   

        
class NORM_net(nn.Module):
    def __init__(self, modes, nodes,Fmodes, width, TIME_MATRIX, TIME_INVERSE, LBO_MATRIX, LBO_INVERSE, Nt):
        super(NORM_net, self).__init__()

        self.modes1 = modes
        self.modes2 = Fmodes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(6, self.width) 
        
        self.TIME_MATRIX = TIME_MATRIX
        self.TIME_INVERSE = TIME_INVERSE
        self.LBO_MATRIX = LBO_MATRIX 
        self.LBO_INVERSE = LBO_INVERSE 
        self.Nx = LBO_MATRIX.size(0)
        self.Nt = Nt     
        self.nodes = nodes

        self.convt = Spatiotemporal_Parameterization(self.nodes, self.nodes, self.width)
        
        self.conv0 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX,self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)  
        self.conv1 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2,self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)
        self.conv2 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv3 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        
        self.conv4 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv5 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv6 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv7 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )

        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x): 
        
        '''
        Input:(n*Nt*6)
        extend channel from 6 to width, obtain n*Nt*width
        '''
        x = self.fc0(x) 
        
        '''
        Prject time-domain to the Frequency-domain, here we can use FFT or 1D LBO
        '''
        x = self.Fmapping_low(x)
        
        '''
        Add a new frequency channel for spatial-domain
        Extend the width of the new channel
        '''
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2],1)
        x = self.Extend(x,self.modes1)
        
        '''
        Project the constructed frequency weight back to original domain
        '''
        x1 = x.permute(3, 1, 0, 2)
        x1 = self.iFmapping(x1, self.Nt)
        x1 = self.iLmapping(x1, self.LBO_MATRIX)
        
        '''
        Parameterize on Spatiotemporal domain to increase the expressiveness of the model
        '''
        x = x1.permute(0,1,3,2)
        x = self.convt(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1, 3)
        

        '''
        layer 1
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        
        x1 = self.conv0(x1)  
        x1 = self.Fmapping(x1, self.modes2)        
        x1 = self.conv4(x1)           
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w0(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 2
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)    
        x1 = self.conv1(x1) 
        x1 = self.Fmapping(x1, self.modes2) 
        x1 = self.conv5(x1) 
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w1(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 3
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)      
        x1 = self.conv2(x1) 
        x1 = self.Fmapping(x1, self.modes2)
        x1 = self.conv6(x1) 
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3)
        x2 = self.w2(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 4
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        x1 = self.conv3(x1)                
        x1 = self.Fmapping(x1, self.modes2)         
        x1 = self.conv7(x1)           
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3)
        x2 = self.w3(x2) 

        x2 = x2.permute(2, 0, 1, 3) 
        x = x1 + x2

        x = x.permute(0, 1, 3, 2)
        x = self.fc1(x) 
        x = torch.relu(x)
        x = self.fc2(x) 
        
        x = x.permute(1, 2, 0, 3)
        
        return x  

    def get_grid(self, shape, device):
        timenodes, batchsize, size_x = shape[2], shape[0], shape[1] 
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([timenodes, batchsize, 1, 1])
        return gridx.to(device)     
    
    def Lmapping(self, x, LBO_MATRIX, LBO_INVERSE):

        x = x = x.permute(0,1,3,2) 
        x = self.LBO_INVERSE @ x  
        x = x.permute(0,1,3,2) 
        return x
        
               
    def Fmapping(self, x, modes2): 

        x = x.permute(1,2,3,0)    
        x_ft = torch.fft.rfft(x)  
        x_ft = x_ft.permute(0,3,1,2) 
        return x_ft
    
    def iFmapping(self, x, Nt):

        x = x.permute(2,3,0,1) 
        
        x_rft = torch.fft.irfft(x, Nt)  
        
        return x_rft
        
    def iLmapping(self, x, LBO_MATRIX): 

        x = x.permute(3,0,1,2) 
        
        x = x @ LBO_MATRIX.T 
        
        return x
    
    def Fmapping_low(self, x): 

        x = x.permute(0,2,1)  
        
        x_ft = torch.fft.rfft(x) 
        
        x_ft = x_ft.permute(0,2,1) 
        
        return x_ft
    
    def Extend(self, x, modes): 

        scale = (1 / (x.shape[2] * modes))
        weights1 = nn.Parameter(scale*torch.rand(x.shape[1], x.shape[2], modes, dtype=torch.float)).cuda()
        x = torch.einsum("txbi,xio->txbo", x, weights1)
        
        return x
        
class NORM_net_DeltaPhi(nn.Module):
    def __init__(self, modes, nodes,Fmodes, width, TIME_MATRIX, TIME_INVERSE, LBO_MATRIX, LBO_INVERSE, Nt):
        super(NORM_net_DeltaPhi, self).__init__()

        self.modes1 = modes
        self.modes2 = Fmodes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(22, self.width) 
        
        self.TIME_MATRIX = TIME_MATRIX
        self.TIME_INVERSE = TIME_INVERSE
        self.LBO_MATRIX = LBO_MATRIX 
        self.LBO_INVERSE = LBO_INVERSE 
        self.Nx = LBO_MATRIX.size(0)
        self.Nt = Nt     
        self.nodes = nodes

        self.convt = Spatiotemporal_Parameterization(self.nodes, self.nodes, self.width)
        
        self.conv0 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX,self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)  
        self.conv1 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2,self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)
        self.conv2 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv3 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        
        self.conv4 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv5 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv6 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv7 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )

        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']
        # print(ref_y.shape)
        # exit()
        ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(x)

        grid = self.get_grid(x)
        # print(x.shape, ref_score.shape, ref_x.shape, grid.shape)
        ref_y_2 = ref_y[:, -1, :, :]   # → [B, N, C]
        x = torch.cat((x, ref_score, ref_x, ref_y_2, grid), dim=-1)

        x = self.fc0(x)
        # print("x:",x.shape)#[10,121,16]
        x = self.Fmapping_low(x)
        
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        # print("x:",x.shape)#[10,61,16,1]
        x = self.Extend(x, self.modes1)
        # exit()
        x1 = x.permute(3, 1, 0, 2)
        x1 = self.iFmapping(x1, self.Nt)
        x1 = self.iLmapping(x1, self.LBO_MATRIX)

        x = x1.permute(0, 1, 3, 2)
        x = self.convt(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1, 3)    

        '''
        layer 1
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        
        x1 = self.conv0(x1)  
        x1 = self.Fmapping(x1, self.modes2)        
        x1 = self.conv4(x1)           
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w0(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 2
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)    
        x1 = self.conv1(x1) 
        x1 = self.Fmapping(x1, self.modes2) 
        x1 = self.conv5(x1) 
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w1(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 3
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)      
        x1 = self.conv2(x1) 
        x1 = self.Fmapping(x1, self.modes2)
        x1 = self.conv6(x1) 
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3)
        x2 = self.w2(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 4
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        x1 = self.conv3(x1)                
        x1 = self.Fmapping(x1, self.modes2)         
        x1 = self.conv7(x1)           
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3)
        x2 = self.w3(x2) 

        x2 = x2.permute(2, 0, 1, 3) 
        x = x1 + x2

        x = x.permute(0, 1, 3, 2)
        x = self.fc1(x) 
        x = torch.relu(x)
        x = self.fc2(x) 
        
        x = x.permute(1, 2, 0, 3)
        # print(ref_y.shape)
        return x + ref_y.reshape(x.shape) 
        # return x

    def get_grid(self, x):
        B, N, C = x.shape     # batch, nodes, channels
        device = x.device
        grid = torch.linspace(0, 1, N, device=device)       # [N]
        grid = grid.unsqueeze(0).expand(B, N)               # [B, N]
        grid = grid.unsqueeze(-1)                           # [B, N, 1]        
        return grid
 
    def Lmapping(self, x, LBO_MATRIX, LBO_INVERSE):

        x = x = x.permute(0,1,3,2) 
        x = self.LBO_INVERSE @ x  
        x = x.permute(0,1,3,2) 
        return x
        
               
    def Fmapping(self, x, modes2): 

        x = x.permute(1,2,3,0)    
        x_ft = torch.fft.rfft(x)  
        x_ft = x_ft.permute(0,3,1,2) 
        return x_ft
    
    def iFmapping(self, x, Nt):

        x = x.permute(2,3,0,1) 
        
        x_rft = torch.fft.irfft(x, Nt)  
        
        return x_rft
        
    def iLmapping(self, x, LBO_MATRIX): 

        x = x.permute(3,0,1,2) 
        
        x = x @ LBO_MATRIX.T 
        
        return x
    
    def Fmapping_low(self, x): 

        x = x.permute(0,2,1)  
        
        x_ft = torch.fft.rfft(x) 
        
        x_ft = x_ft.permute(0,2,1) 
        
        return x_ft
    
    def Extend(self, x, modes): 

        scale = (1 / (x.shape[2] * modes))
        weights1 = nn.Parameter(scale*torch.rand(x.shape[1], x.shape[2], modes, dtype=torch.float)).cuda()
        x = torch.einsum("txbi,xio->txbo", x, weights1)
        
        return x

# class NORM_net_DeltaPhi_ODE(nn.Module):
#     def __init__(self, modes, nodes, Fmodes, width, TIME_MATRIX, TIME_INVERSE, LBO_MATRIX, LBO_INVERSE, Nt, steps=4, coord_dim=6):
#         super(NORM_net_DeltaPhi_ODE, self).__init__()

#         self.modes1 = modes
#         self.modes2 = Fmodes
#         self.width = width
#         self.padding = 2 
#         self.fc0 = nn.Linear(5, self.width) 
#         self.fc3 = nn.Linear(12, self.width) 
        
#         self.TIME_MATRIX = TIME_MATRIX
#         self.TIME_INVERSE = TIME_INVERSE
#         self.LBO_MATRIX = LBO_MATRIX 
#         self.LBO_INVERSE = LBO_INVERSE 
#         self.Nx = LBO_MATRIX.size(0)
#         self.Nt = Nt     
#         self.nodes = nodes

#         self.convt = Spatiotemporal_Parameterization(self.nodes, self.nodes, self.width)
        
#         self.conv0 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX,self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)  
#         self.conv1 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2,self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)
#         self.conv2 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
#         self.conv3 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        
#         self.conv4 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
#         self.conv5 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
#         self.conv6 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
#         self.conv7 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )

#         self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
#         self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
#         self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
#         self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 3)
#         self.steps = steps
#         self.coord_proj = nn.Linear(coord_dim, self.width)
#         self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
#         self.mode_proj = nn.Linear(121, self.width)  # 121 -> 64
#         self.dhdt_expand = nn.Conv1d(self.width, 2*self.width, 1)
#         self.proj = nn.Linear(3, 16)

#     def forward(self, x):
#         x_coord, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

#         ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(ref_x)
#         B, N = x['x'].shape[0], x['x'].shape[1]
#         ref_score = x['ref_score'].view(B, 1, 1).repeat(1, N, 1)

#         grid = self.get_grid(x_coord)
#         ref_y = x['ref_y']
#         if ref_y.dim() == 2:
#             ref_y = ref_y.unsqueeze(-1)
#         ref_y = ref_y[:, -1, :, :]   # → [B, N, C]

#         grid = self.get_grid(torch.zeros(B, N, 1).to(device))
#         # print(ref_score.shape,ref_y.shape,grid.shape)
#         x_in = torch.cat([ref_score, ref_y, grid], dim=-1)   # (B, N, 3)
        
#         x_feat = self.fc0(x_in)   # (B, N, width)

#         a_input = torch.cat([ref_x, x_coord], dim=-1)   # (B, N, 2)
#         a_feat = self.fc3(a_input)                     # (B, N, width)
#         # a = a_feat

#         depth_coord = (x_coord - ref_x) / float(self.steps)   # (B, N, coord_dim)
#         depth_feat = self.coord_proj(depth_coord)             # (B, N, width)
#         # depth_scale = depth_feat.permute(0, 2, 1).contiguous()  # (B, width, N)
        
#         h = x_feat.permute(0, 2, 1).contiguous()    # (B, width, N)
#         # a = a_feat.permute(0, 2, 1).contiguous()    # (B, width, N)

#         target_N = h.shape[1]

#         a = a_feat.permute(0,2,1)
#         depth_scale = depth_feat.permute(0,2,1)
#         # print(h.shape,depth_scale.shape)

#         for _ in range(self.steps):
#             k1 = self.func(a, h)
#             # print(a.shape, depth_scale.shape, h.shape, k1.shape)
#             k2 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k1)
#             k3 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k2)
#             k4 = self.func(a + depth_scale, h + depth_scale * k3)
#             h = h + (depth_scale / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

#         x_out = h.permute(0, 2, 1).contiguous()   # (B, N, width)
#         x_out = F.gelu(self.fc1(x_out))            # (B, N, 128)
#         x_out = self.fc2(x_out)                    # (B, N, 1)

#         return x_out + ref_y.reshape(x_out.shape)  # (B, N, 1)
#     def func(self, a, h):
#         ha = torch.cat([h, a], dim=1)   # (B, 2*width, N)
#         # print(h.shape, self.width)
#         x = self.Fmapping_low(ha)
#         # print(x.shape, self.nodes, self.width)

#         x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
#         x = self.Extend(x, self.modes1)

#         x1 = x.permute(3, 1, 0, 2)
#         x1 = self.iFmapping(x1, self.Nt)
#         x1 = self.iLmapping(x1, self.LBO_MATRIX)
#         x = x1.permute(0, 1, 3, 2)
#         t, b, i, xdim = x.shape  # = 121,10,1656,121
#         x = x.reshape(-1, xdim)  # -> (t*b*i, xdim)
#         x = self.mode_proj(x)    # -> (t*b*i, modes1)
#         x = x.view(t, b, i, self.width)  # -> [121, 10, 1656, 64]
#         x = self.convt(x)
#         x = torch.relu(x)
#         x = x.permute(0, 2, 1, 3)   

#         '''
#         layer 1
#         '''
#         x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        
#         x1 = self.conv0(x1)  
#         x1 = self.Fmapping(x1, self.modes2)        
#         x1 = self.conv4(x1)           
#         x1 = self.iFmapping(x1, self.Nt) 
#         x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
#         x2 = x.permute(1, 2, 0, 3) 
#         x2 = self.w0(x2) 
#         x2 = x2.permute(2, 0, 1, 3) 
        
#         x = x1 + x2
#         x = torch.relu(x) 
        
#         '''
#         layer 2
#         '''
#         x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)    
#         x1 = self.conv1(x1) 
#         x1 = self.Fmapping(x1, self.modes2) 
#         x1 = self.conv5(x1) 
#         x1 = self.iFmapping(x1, self.Nt) 
#         x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
#         x2 = x.permute(1, 2, 0, 3) 
#         x2 = self.w1(x2) 
#         x2 = x2.permute(2, 0, 1, 3) 
        
#         x = x1 + x2
#         x = torch.relu(x) 
        
#         '''
#         layer 3
#         '''
#         x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)      
#         x1 = self.conv2(x1) 
#         x1 = self.Fmapping(x1, self.modes2)
#         x1 = self.conv6(x1) 
#         x1 = self.iFmapping(x1, self.Nt) 
#         x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
#         x2 = x.permute(1, 2, 0, 3)
#         x2 = self.w2(x2) 
#         x2 = x2.permute(2, 0, 1, 3) 
        
#         x = x1 + x2
#         x = torch.relu(x) 
        
#         '''
#         layer 4
#         '''
#         x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
#         x1 = self.conv3(x1)                
#         x1 = self.Fmapping(x1, self.modes2)         
#         x1 = self.conv7(x1)           
#         x1 = self.iFmapping(x1, self.Nt) 
#         x1 = self.iLmapping(x1, self.LBO_MATRIX) 

#         x2 = x.permute(1, 2, 0, 3)
#         x2 = self.w3(x2) 

#         x2 = x2.permute(2, 0, 1, 3) 
#         x = x1 + x2

#         x = x.permute(0, 1, 3, 2)
#         x = self.fc1(x) 
#         x = torch.relu(x)
#         x = self.fc2(x) 
        
#         x = x.permute(1, 2, 0, 3)
#         # print(x.shape)
#         x = self.proj(x)          # (B, 1656, 121, 16)
#         x = x.mean(dim=1)         # (B, 121, 16)
#         dhdt_h = x.permute(0, 2, 1)    # (B, 16, 121)

#         return dhdt_h

#     def get_grid(self, x):
#         B, N, C = x.shape     # batch, nodes, channels
#         device = x.device
#         grid = torch.linspace(0, 1, N, device=device)       # [N]
#         grid = grid.unsqueeze(0).expand(B, N)               # [B, N]
#         grid = grid.unsqueeze(-1)                           # [B, N, 1]        
#         return grid
 
#     def Lmapping(self, x, LBO_MATRIX, LBO_INVERSE):

#         x = x = x.permute(0,1,3,2) 
#         x = self.LBO_INVERSE @ x  
#         x = x.permute(0,1,3,2) 
#         return x
        
               
#     def Fmapping(self, x, modes2): 

#         x = x.permute(1,2,3,0)    
#         x_ft = torch.fft.rfft(x)  
#         x_ft = x_ft.permute(0,3,1,2) 
#         return x_ft
    
#     def iFmapping(self, x, Nt):

#         x = x.permute(2,3,0,1) 
        
#         x_rft = torch.fft.irfft(x, Nt)  
        
#         return x_rft
        
#     def iLmapping(self, x, LBO_MATRIX): 

#         x = x.permute(3,0,1,2) 
        
#         x = x @ LBO_MATRIX.T 
        
#         return x
    
#     def Fmapping_low(self, x): 

#         x = x.permute(0,2,1)  
        
#         x_ft = torch.fft.rfft(x) 
        
#         x_ft = x_ft.permute(0,2,1) 
        
#         return x_ft
    
#     def Extend(self, x, modes): 

#         scale = (1 / (x.shape[2] * modes))
#         weights1 = nn.Parameter(scale*torch.rand(x.shape[1], x.shape[2], modes, dtype=torch.float)).cuda()
#         x = torch.einsum("txbi,xio->txbo", x, weights1)
        
#         return x
# import torch.nn.functional as F

# class NORM_net_DeltaPhi_ODE(nn.Module):
#     def __init__(self, modes, nodes,Fmodes, width, TIME_MATRIX, TIME_INVERSE, LBO_MATRIX, LBO_INVERSE, Nt, steps=4, coord_dim=6):
#         super(NORM_net_DeltaPhi_ODE, self).__init__()

#         self.modes1 = modes
#         self.modes2 = Fmodes
#         self.width = width
#         self.padding = 2 
#         self.fc0 = nn.Linear(5, self.width) 
#         self.fc4 = nn.Linear(self.width, self.width)
        
#         self.TIME_MATRIX = TIME_MATRIX
#         self.TIME_INVERSE = TIME_INVERSE
#         self.LBO_MATRIX = LBO_MATRIX 
#         self.LBO_INVERSE = LBO_INVERSE 
#         self.Nx = LBO_MATRIX.size(0)
#         self.Nt = Nt     
#         self.nodes = nodes

#         self.convt = Spatiotemporal_Parameterization(self.nodes, self.nodes, self.width)
        
#         self.conv0 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX,self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)  
#         self.conv1 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2,self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)
#         self.conv2 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
#         self.conv3 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        
#         self.conv4 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
#         self.conv5 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
#         self.conv6 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
#         self.conv7 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )

#         self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
#         self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
#         self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
#         self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 3)
#         self.fc3 = nn.Linear(12, self.width)

#         self.dhdt_expand = nn.Conv1d(self.width, 2*self.width, 1)
#         self.coord_proj = nn.Linear(coord_dim, self.width)

#         self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
#         self.steps = steps   
#         self.coord_dim = coord_dim
#         # self.ha_conv = nn.Conv1d(121, 10, 1)
#         self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
#         self.weights1 = nn.Parameter(torch.rand(self.nodes, self.width, self.width).float())  # 示例 shape
#         self.extend_weights = nn.Parameter(torch.rand(self.width, self.width))


#     def func(self, a, h):
#         print("a.shape:",a.shape,"h.shape:",h.shape)
#         ha = torch.cat([h, a], dim=1)   # (B, 2*width, N)
#         ha = self.ha_conv(ha)           # (B, width, N)
#         h = F.gelu(ha)
#         # print(x.shape, ref_score.shape, ref_x.shape, grid.shape)
#         # x = torch.cat((x, ref_score, ref_x, grid), dim=-1)

#         # x = self.fc0(ha)
#         # 调整shape再送入 fc0
#         print("h:",h.shape)
#         # 调整 shape 后 fc
#         ha_perm = ha.permute(0, 2, 1).contiguous()  # [B, N, width]
#         x = self.fc4(ha_perm)                        # [B, N, width]
#         x = x.permute(0, 2, 1).contiguous()          # [B, width, N]
#         print("x.shape:",x.shape)
#         # 低频 fft
#         x = self.Fmapping_low(x)                     # [B, width, N]
#         print("x.shape:",x.shape)
#         # 扩展维度，使用 extend_weights
#         x = self.Extend(x, self.modes1)                            # [B, width, N]

#         '''
#         layer 1
#         '''
#         print("x.shape:",x.shape)
#         x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        
#         x1 = self.conv0(x1)  
#         x1 = self.Fmapping(x1, self.modes2)        
#         x1 = self.conv4(x1)           
#         x1 = self.iFmapping(x1, self.Nt) 
#         x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
#         x2 = x.permute(1, 2, 0, 3) 
#         x2 = self.w0(x2) 
#         x2 = x2.permute(2, 0, 1, 3) 
        
#         x = x1 + x2
#         x = torch.relu(x) 
        
#         '''
#         layer 2
#         '''
#         x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)    
#         x1 = self.conv1(x1) 
#         x1 = self.Fmapping(x1, self.modes2) 
#         x1 = self.conv5(x1) 
#         x1 = self.iFmapping(x1, self.Nt) 
#         x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
#         x2 = x.permute(1, 2, 0, 3) 
#         x2 = self.w1(x2) 
#         x2 = x2.permute(2, 0, 1, 3) 
        
#         x = x1 + x2
#         x = torch.relu(x) 
        
#         '''
#         layer 3
#         '''
#         x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)      
#         x1 = self.conv2(x1) 
#         x1 = self.Fmapping(x1, self.modes2)
#         x1 = self.conv6(x1) 
#         x1 = self.iFmapping(x1, self.Nt) 
#         x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
#         x2 = x.permute(1, 2, 0, 3)
#         x2 = self.w2(x2) 
#         x2 = x2.permute(2, 0, 1, 3) 
        
#         x = x1 + x2
#         x = torch.relu(x) 
        
#         '''
#         layer 4
#         '''
#         x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
#         x1 = self.conv3(x1)                
#         x1 = self.Fmapping(x1, self.modes2)         
#         x1 = self.conv7(x1)           
#         x1 = self.iFmapping(x1, self.Nt) 
#         x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
#         x2 = x.permute(1, 2, 0, 3)
#         x2 = self.w3(x2) 

#         x2 = x2.permute(2, 0, 1, 3) 
#         x = x1 + x2

#         x = x.permute(0, 1, 3, 2)
#         x = self.fc1(x) 
#         x = torch.relu(x)
#         x = self.fc2(x) 
        
#         x = x.permute(1, 2, 0, 3)


#         dhdt = self.dhdt_expand(x)  # (B, 2*width, N)
#         dhdt_h, dhdt_a = torch.split(dhdt, self.width, dim=1)

#         return dhdt_h
#     # def forward(self, x):
#     #     x_coord, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']
#     #     print("x_coord.shape:",x_coord.shape,"ref_y.shape:",ref_y.shape)
#     #     ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(x_coord)

#     #     grid = self.get_grid(x_coord)
#     #     # print("ref_score:",ref_score.shape,",ref_y:",ref_y.shape,",grid:",grid.shape)
#     #     # x_in = torch.cat([ref_score, ref_y, grid], dim=-1)   # (B, N, 3)
#     #     # 假设你只想取第0个索引的1656维
#     #     ref_y_flat = ref_y[:, -1, :, :]  # [10, 121, 3]
#     #     x_in = torch.cat([ref_score, ref_y_flat, grid], dim=-1)  # [10, 121, 10] (6+3+1)

#     #     x_feat = self.fc0(x_in)   # (B, N, width)

#     #     a_input = torch.cat([ref_x, x_coord], dim=-1)   # (B, N, 2)
#     #     a_feat = self.fc3(a_input)                     # (B, N, width)

#     #     depth_coord = (x_coord - ref_x) / float(self.steps)   # (B, N, coord_dim)
#     #     depth_feat = self.coord_proj(depth_coord)             # (B, N, width)
#     #     depth_scale = depth_feat.permute(0, 2, 1).contiguous()  # (B, width, N)
        
#     #     h = x_feat.permute(0, 2, 1).contiguous()    # (B, width, N)
#     #     a = a_feat.permute(0, 2, 1).contiguous()    # (B, width, N)

#     #     for _ in range(self.steps):
#     #         k1 = self.func(a, h)
#     #         k2 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k1)
#     #         k3 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k2)
#     #         k4 = self.func(a + depth_scale, h + depth_scale * k3)
#     #         h = h + (depth_scale / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

#     #     x_out = h.permute(0, 2, 1).contiguous()   # (B, N, width)
#     #     x_out = F.gelu(self.fc1(x_out))            # (B, N, 128)
#     #     x_out = self.fc2(x_out)                    # (B, N, 1)

#     #     return x_out + ref_y.reshape(x_out.shape) 
        
#     # def interpolate_nodes(self, x, target_N):
#     #     """
#     #     将 x [B, F, N] 或 [B, N, F] 插值到 target_N 个节点
#     #     """
#     #     if x.dim() == 3:
#     #         # [B, F, N] -> [B, N, F] 方便 interpolate
#     #         x_perm = x.permute(0, 2, 1)
#     #         x_interp = F.interpolate(x_perm, size=target_N, mode='linear', align_corners=True)
#     #         x_out = x_interp.permute(0, 2, 1)
#     #         return x_out
#     #     elif x.dim() == 2:
#     #         # [B, N]
#     #         x = x.unsqueeze(1)  # [B, 1, N]
#     #         x_interp = F.interpolate(x, size=target_N, mode='linear', align_corners=True)
#     #         return x_interp.squeeze(1)
#     #     else:
#     #         raise ValueError(f"Unsupported tensor shape {x.shape}")

#     # forward 修改
#     def forward(self, x):
#         x_coord, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']
#         B = x_coord.size(0)
#         target_N = ref_y.size(1)  # 1656

#         # 插值 x_coord 和 ref_x 到 1656 节点
        
#         # x_coord_interp = self.interpolate_nodes(x_coord.permute(0, 2, 1), target_N).permute(0, 2, 1)  # [B, target_N, 6]
#         # ref_x_interp = self.interpolate_nodes(ref_x.permute(0, 2, 1), target_N).permute(0, 2, 1)      # [B, target_N, 6]
#         # print("x_coord_interp:",x_coord_interp.shape,"ref_x_interp",ref_x_interp.shape)
#         # x_coord_perm = x_coord.permute(0, 2, 1)  # [B, coord_dim=6, N=121]
#         # x_coord_interp = F.interpolate(x_coord_perm, size=target_N, mode='linear', align_corners=True)
#         # x_coord_interp = x_coord_interp.permute(0, 2, 1)  # [B, 1656, 6]

#         # ref_x_perm = ref_x.permute(0, 2, 1)
#         # ref_x_interp = F.interpolate(ref_x_perm, size=target_N, mode='linear', align_corners=True)
#         # ref_x_interp = ref_x_interp.permute(0, 2, 1)      # [B, 1656, 6]
#         # print("x_coord_interp:",x_coord_interp.shape,"ref_x_interp",ref_x_interp.shape)

#         # ref_score = ref_score.reshape(B, 1, 1) * torch.ones_like(x_coord_interp)

#         grid = self.get_grid(x_coord_interp)
#         ref_score_interp = ref_score.reshape(B, 1, 1) * torch.ones((B, target_N, 1), device=x_coord.device)
#         print("ref_score_interp:",ref_score_interp.shape,"grid:",grid.shape)

#         # 取最后时间步 ref_y
#         ref_y_flat = ref_y[:, :, -1, :]  # [B, 1656, 3]
#         print("ref_y_flat:",ref_y_flat.shape)
#         # x_in = torch.cat([ref_score, ref_y_flat, grid], dim=-1)  # [B, 1656, 10]
#         x_in = torch.cat([ref_score_interp, ref_y_flat, grid], dim=-1)  # [B, 1656, 5]
#         print("x_in:",x_in.shape)
#         x_feat = self.fc0(x_in)   # [B, 1656, width]
#         a_input = torch.cat([ref_x_interp, x_coord_interp], dim=-1)  # [B, 1656, 12]
#         a_feat = self.fc3(a_input)                                     # [B, 1656, width]

#         depth_coord = (x_coord_interp - ref_x_interp) / float(self.steps)
#         depth_feat = self.coord_proj(depth_coord)
#         depth_scale = depth_feat.permute(0, 2, 1).contiguous()        # [B, width, 1656]

#         h = x_feat.permute(0, 2, 1).contiguous()                      # [B, width, 1656]
#         a = a_feat.permute(0, 2, 1).contiguous()                      # [B, width, 1656]

#         # RK4
#         for _ in range(self.steps):
#             k1 = self.func(a, h)
#             k2 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k1)
#             k3 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k2)
#             k4 = self.func(a + depth_scale, h + depth_scale * k3)
#             h = h + (depth_scale / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

#         x_out = h.permute(0, 2, 1).contiguous()   # [B, 1656, width]
#         x_out = F.gelu(self.fc1(x_out))
#         x_out = self.fc2(x_out)                    # [B, 1656, 3]

#         return x_out + ref_y_flat  # 与 ref_y 对齐


#     def get_grid(self, x):
#         B, N, C = x.shape     # batch, nodes, channels
#         device = x.device
#         grid = torch.linspace(0, 1, N, device=device)       # [N]
#         grid = grid.unsqueeze(0).expand(B, N)               # [B, N]
#         grid = grid.unsqueeze(-1)                           # [B, N, 1]        
#         return grid
 
#     def Lmapping(self, x, LBO_MATRIX, LBO_INVERSE):
#         if x.dim() == 4:  # [B, ?, ?, ?]
#             x = x.permute(0,1,3,2)
#             x = LBO_INVERSE @ x
#             x = x.permute(0,1,3,2)
#         elif x.dim() == 3:  # [B, F, N]
#             x = x @ LBO_INVERSE.T  # 直接矩阵乘
#         else:
#             raise ValueError(f"Unsupported tensor shape {x.shape}")
#         return x

        
               
#     def Fmapping(self, x, modes2): 

#         x = x.permute(1,2,3,0)    
#         x_ft = torch.fft.rfft(x)  
#         x_ft = x_ft.permute(0,3,1,2) 
#         return x_ft
    
#     def iFmapping(self, x, Nt):

#         x = x.permute(2,3,0,1) 
        
#         x_rft = torch.fft.irfft(x, Nt)  
        
#         return x_rft
        
#     def iLmapping(self, x, LBO_MATRIX): 

#         x = x.permute(3,0,1,2) 
        
#         x = x @ LBO_MATRIX.T 
        
#         return x
    
#     # def Extend(self, x):
#     #     """
#     #     将 x [B, F_in, N] -> [B, F_out, N]，使用可训练矩阵
#     #     """
#     #     if torch.is_complex(x):
#     #         x = x.real   # 取实部
#     #     B, F_in, N = x.shape
#     #     # matmul: [B, N, F_in] x [F_in, F_out] -> [B, N, F_out]
#     #     x_perm = x.permute(0, 2, 1)                       # [B, N, F_in]
#     #     x_ext = torch.matmul(x_perm, self.extend_weights)  # [B, N, F_out]
#     #     x_ext = x_ext.permute(0, 2, 1).contiguous()       # [B, F_out, N]
#     #     return x_ext

#     # def Fmapping_low(self, x):
#     #     # 低频 fft
#     #     x = x.permute(0, 2, 1)  # [B, N, width]
#     #     x_ft = torch.fft.rfft(x, dim=1)  # [B, N_rfft, width]
#     #     x_ft = x_ft.permute(0, 2, 1).contiguous()
#     #     return x_ft
#     def Fmapping_low(self, x): 

#         x = x.permute(0,2,1)  
        
#         x_ft = torch.fft.rfft(x) 
        
#         x_ft = x_ft.permute(0,2,1) 
        
#         return x_ft
    
#     def Extend(self, x, modes): 

#         scale = (1 / (x.shape[2] * modes))
#         weights1 = nn.Parameter(scale*torch.rand(x.shape[1], x.shape[2], modes, dtype=torch.float)).cuda()
#         x = torch.einsum("txbi,xio->txbo", x, weights1)
        
#         return x

class NORM_net_DeltaPhi_ODE(nn.Module):
    def __init__(self, modes, nodes,Fmodes, width, TIME_MATRIX, TIME_INVERSE, LBO_MATRIX, LBO_INVERSE, Nt, steps=10, coord_dim=6):
        super(NORM_net_DeltaPhi_ODE, self).__init__()

        self.modes1 = modes
        self.modes2 = Fmodes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(10, self.width) 
        self.fc4 = nn.Linear(self.width, self.width)
        
        self.TIME_MATRIX = TIME_MATRIX
        self.TIME_INVERSE = TIME_INVERSE
        self.LBO_MATRIX = LBO_MATRIX 
        self.LBO_INVERSE = LBO_INVERSE 
        self.Nx = LBO_MATRIX.size(0)
        self.Nt = Nt     
        self.nodes = nodes

        self.convt = Spatiotemporal_Parameterization(self.nodes, self.nodes, self.width)
        
        self.conv0 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX,self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)  
        self.conv1 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2,self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)
        self.conv2 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv3 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        
        self.conv4 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv5 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv6 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv7 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )

        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)
        self.fc3 = nn.Linear(12, self.width)

        self.dhdt_expand = nn.Conv1d(self.width, 2*self.width, 1)
        self.coord_proj = nn.Linear(coord_dim, self.width)

        self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
        self.steps = steps   
        self.coord_dim = coord_dim
        # self.ha_conv = nn.Conv1d(121, 10, 1)
        self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
        self.weights1 = nn.Parameter(torch.rand(self.nodes, self.width, self.width).float())  # 示例 shape

    def func(self, a, h):
        ha = torch.cat([h, a], dim=1)   # (B, 2*width, N)
        ha = self.ha_conv(ha)           # (B, width, N)
        h = F.gelu(ha)
        # print(x.shape, ref_score.shape, ref_x.shape, grid.shape)
        # x = torch.cat((x, ref_score, ref_x, grid), dim=-1)

        # x = self.fc0(ha)
        # 调整shape再送入 fc0
        ha = ha.permute(0, 2, 1).contiguous()  # [B, N, width]
        x = self.fc4(ha)                       # [B, N, width]
        # x = x.permute(0, 2, 1).contiguous()    # [B, width, N] 恢复原形状

        # print("x:",x.shape)#[10,121,16]
        x = self.Fmapping_low(x)

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        # print("x:",x.shape)
        x = self.Extend(x, self.modes1)

        '''
        Project the constructed frequency weight back to original domain
        '''
        x1 = x.permute(3, 1, 0, 2)
        x1 = self.iFmapping(x1, self.Nt)
        x1 = self.iLmapping(x1, self.LBO_MATRIX)
        
        '''
        Parameterize on Spatiotemporal domain to increase the expressiveness of the model
        '''
        x = x1.permute(0,1,3,2)
        x = self.convt(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1, 3)
        print("x:",x.shape)#x: torch.Size([121, 10, 16, 1656])
        # exit()
        '''
        layer 1
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        
        x1 = self.conv0(x1)  
        x1 = self.Fmapping(x1, self.modes2)        
        x1 = self.conv4(x1)           
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w0(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 2
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)    
        x1 = self.conv1(x1) 
        x1 = self.Fmapping(x1, self.modes2) 
        x1 = self.conv5(x1) 
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w1(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 3
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)      
        x1 = self.conv2(x1) 
        x1 = self.Fmapping(x1, self.modes2)
        x1 = self.conv6(x1) 
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3)
        x2 = self.w2(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 4
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        x1 = self.conv3(x1)                
        x1 = self.Fmapping(x1, self.modes2)         
        x1 = self.conv7(x1)           
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3)
        x2 = self.w3(x2) 

        x2 = x2.permute(2, 0, 1, 3) 
        x = x1 + x2
        # print(x.shape)# torch.Size([121, 10, 16, 1656])
        # exit()
        x = x.permute(0, 1, 3, 2)
        x = self.fc1(x) 
        x = torch.relu(x)
        x = self.fc2(x) 
        
        x = x.permute(1, 2, 0, 3)
        # x = x.mean(dim=-1)    # 平均 D 维 -> [121, 10, 16]
        # x = x.permute(1, 2, 0)  # -> [10, 16, 121]


        # dhdt = self.dhdt_expand(x)  # (B, 2*width, N)
        # dhdt_h, dhdt_a = torch.split(dhdt, self.width, dim=1)

        # return dhdt_h
        return x
    def forward(self, x):
        x_coord, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

        ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(x_coord)

        grid = self.get_grid(x_coord)
        # print("ref_score:",ref_score.shape,",ref_y:",ref_y.shape,",grid:",grid.shape)
        # x_in = torch.cat([ref_score, ref_y, grid], dim=-1)   # (B, N, 3)
        # 假设你只想取第0个索引的1656维
        ref_y_flat = ref_y[:, -1, :, :]  # [10, 121, 3]
        x_in = torch.cat([ref_score, ref_y_flat, grid], dim=-1)  # [10, 121, 10] (6+3+1)

        x_feat = self.fc0(x_in)   # (B, N, width)

        a_input = torch.cat([ref_x, x_coord], dim=-1)   # (B, N, 2)
        a_feat = self.fc3(a_input)                     # (B, N, width)

        depth_coord = (x_coord - ref_x) / float(self.steps)   # (B, N, coord_dim)
        depth_feat = self.coord_proj(depth_coord)             # (B, N, width)
        depth_scale = depth_feat.permute(0, 2, 1).contiguous()  # (B, width, N)
        
        h = x_feat.permute(0, 2, 1).contiguous()    # (10, 16, 121)
        a = a_feat.permute(0, 2, 1).contiguous()    # (10, 16, 121)

        for _ in range(self.steps):
            k1 = self.func(a, h)#[10,16,121]
            # print("k1:",k1.shape,",depth_scale:",depth_scale.shape,"h:",h.shape,",a:",a.shape)
            # exit()
            # k1: torch.Size([10, 1656, 121, 3]) ,depth_scale: torch.Size([10, 16, 121]) h: torch.Size([10, 16, 121]) ,a: torch.Size([10, 16, 121])
            k2 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k1)
            k3 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k2)
            k4 = self.func(a + depth_scale, h + depth_scale * k3)
            h = h + (depth_scale / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x_out = h.permute(0, 2, 1).contiguous()   # (B, N, width)
        x_out = F.gelu(self.fc1(x_out))            # (B, N, 128)
        x_out = self.fc2(x_out)                    # (B, N, 1)

        # return x_out + ref_y.reshape(x_out.shape) 
        return x_out

    def get_grid(self, x):
        B, N, C = x.shape     # batch, nodes, channels
        device = x.device
        grid = torch.linspace(0, 1, N, device=device)       # [N]
        grid = grid.unsqueeze(0).expand(B, N)               # [B, N]
        grid = grid.unsqueeze(-1)                           # [B, N, 1]        
        return grid
 
    def Lmapping(self, x, LBO_MATRIX, LBO_INVERSE):

        x = x = x.permute(0,1,3,2) 
        x = self.LBO_INVERSE @ x  
        x = x.permute(0,1,3,2) 
        return x
        
               
    def Fmapping(self, x, modes2): 

        x = x.permute(1,2,3,0)    
        x_ft = torch.fft.rfft(x)  
        x_ft = x_ft.permute(0,3,1,2) 
        return x_ft
    
    def iFmapping(self, x, Nt):

        x = x.permute(2,3,0,1) 
        
        x_rft = torch.fft.irfft(x, Nt)  
        
        return x_rft
        
    def iLmapping(self, x, LBO_MATRIX): 

        x = x.permute(3,0,1,2) 
        
        x = x @ LBO_MATRIX.T 
        
        return x
    
    def Fmapping_low(self, x): 

        x = x.permute(0,2,1)  
        
        x_ft = torch.fft.rfft(x) 
        
        x_ft = x_ft.permute(0,2,1) 
        
        return x_ft
    
    def Extend(self, x, modes): 

        scale = (1 / (x.shape[2] * modes))
        weights1 = nn.Parameter(scale*torch.rand(x.shape[1], x.shape[2], modes, dtype=torch.float)).cuda()
        x = torch.einsum("txbi,xio->txbo", x, weights1)
        
        return x


class NORM_net_DeltaPhi_ODE2(nn.Module):
    def __init__(self, modes, nodes,Fmodes, width, TIME_MATRIX, TIME_INVERSE, LBO_MATRIX, LBO_INVERSE, Nt, steps=4, coord_dim=6):
        super(NORM_net_DeltaPhi_ODE2, self).__init__()

        self.modes1 = modes
        self.modes2 = Fmodes
        self.width = width
        self.padding = 2 
        self.fc0 = nn.Linear(11, self.width) 
        self.fc4 = nn.Linear(self.width, self.width)
        
        self.TIME_MATRIX = TIME_MATRIX
        self.TIME_INVERSE = TIME_INVERSE
        self.LBO_MATRIX = LBO_MATRIX 
        self.LBO_INVERSE = LBO_INVERSE 
        self.Nx = LBO_MATRIX.size(0)
        self.Nt = Nt     
        self.nodes = nodes

        self.convt = Spatiotemporal_Parameterization(self.nodes, self.nodes, self.width)
        
        self.conv0 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX,self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)  
        self.conv1 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2,self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE)
        self.conv2 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv3 = Spatial_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        
        self.conv4 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv5 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv6 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )
        self.conv7 = Temporal_Approximation_block(self.width, self.width, self.modes1, self.modes2, self.LBO_MATRIX, self.LBO_INVERSE, self.TIME_MATRIX, self.TIME_INVERSE )

        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)
        self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1, padding = 0)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 3)
        self.fc3 = nn.Linear(14, self.width)

        self.dhdt_expand = nn.Conv1d(self.width, 2*self.width, 1)
        self.coord_proj = nn.Linear(coord_dim, self.width)

        self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
        self.steps = steps   
        self.coord_dim = coord_dim
        # self.ha_conv = nn.Conv1d(121, 10, 1)
        # self.ha_conv = nn.Conv1d(2*self.width, self.width, 1)
        self.ha_linear = nn.Linear(2 * self.width, self.width)
        self.weights1 = nn.Parameter(torch.rand(self.nodes, self.width, self.width).float())  # 示例 shape

    def func(self, a, h):
        # 假设 self.width = 16
        # self.ha_linear = nn.Linear(2 * self.width, self.width)

        # h, a: (B, 10, 16, 1656)

        # 在 width 维 (dim=2) 上拼接 -> (B, 10, 32, 1656)
        ha = torch.cat([h, a], dim=2)  

        # 把 width 维放到最后，让 Linear 在最后一维上做映射
        # (B, 10, 1656, 32)
        ha = ha.permute(0, 1, 3, 2)

        # 线性映射: 32 -> 16  (作用在最后一维)
        # (B, 10, 1656, 16)
        ha = self.ha_linear(ha)

        # GELU
        ha = F.gelu(ha)

        # 如果你想保持原来的 layout: (B, 10, 16, 1656)
        x = ha.permute(0, 1, 3, 2) # [121, 10, 16, 1656]

        # ha = torch.cat([h, a], dim=1)   # (B, 2*width, N)
        # ha = self.ha_conv(ha)           # (B, width, N)
        # h = F.gelu(ha)
        # # print(x.shape, ref_score.shape, ref_x.shape, grid.shape)
        # # x = torch.cat((x, ref_score, ref_x, grid), dim=-1)

        # # x = self.fc0(ha)
        # # 调整shape再送入 fc0
        # ha = ha.permute(0, 2, 1).contiguous()  # [B, N, width]
        # x = self.fc4(ha)                       # [B, N, width]
        # # x = x.permute(0, 2, 1).contiguous()    # [B, width, N] 恢复原形状

        # # print("x:",x.shape)#[10,121,16]
        # x = self.Fmapping_low(x)

        # x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        # # print("x:",x.shape)
        # x = self.Extend(x, self.modes1)

        # '''
        # Project the constructed frequency weight back to original domain
        # '''
        # x1 = x.permute(3, 1, 0, 2)
        # x1 = self.iFmapping(x1, self.Nt)
        # x1 = self.iLmapping(x1, self.LBO_MATRIX)
        
        # '''
        # Parameterize on Spatiotemporal domain to increase the expressiveness of the model
        # '''
        # x = x1.permute(0,1,3,2)
        # x = self.convt(x)
        # x = torch.relu(x)
        # x = x.permute(0, 2, 1, 3)
        # print("x:",x.shape)#x: torch.Size([121, 10, 16, 1656])
        # exit()
        '''
        layer 1
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        
        x1 = self.conv0(x1)  
        x1 = self.Fmapping(x1, self.modes2)        
        x1 = self.conv4(x1)           
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w0(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 2
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)    
        x1 = self.conv1(x1) 
        x1 = self.Fmapping(x1, self.modes2) 
        x1 = self.conv5(x1) 
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3) 
        x2 = self.w1(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 3
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE)      
        x1 = self.conv2(x1) 
        x1 = self.Fmapping(x1, self.modes2)
        x1 = self.conv6(x1) 
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3)
        x2 = self.w2(x2) 
        x2 = x2.permute(2, 0, 1, 3) 
        
        x = x1 + x2
        x = torch.relu(x) 
        
        '''
        layer 4
        '''
        x1 = self.Lmapping(x, self.LBO_MATRIX, self.LBO_INVERSE) 
        x1 = self.conv3(x1)                
        x1 = self.Fmapping(x1, self.modes2)         
        x1 = self.conv7(x1)           
        x1 = self.iFmapping(x1, self.Nt) 
        x1 = self.iLmapping(x1, self.LBO_MATRIX) 
        
        x2 = x.permute(1, 2, 0, 3)
        x2 = self.w3(x2) 

        x2 = x2.permute(2, 0, 1, 3) 
        x = x1 + x2
        # print(x.shape)# torch.Size([121, 10, 16, 1656])
        # exit()
        # x = x.permute(0, 1, 3, 2)
        # x = self.fc1(x) 
        # x = torch.relu(x)
        # x = self.fc2(x) 
        
        # x = x.permute(1, 2, 0, 3)
        # x = x.mean(dim=-1)    # 平均 D 维 -> [121, 10, 16]
        # x = x.permute(1, 2, 0)  # -> [10, 16, 121]


        # dhdt = self.dhdt_expand(x)  # (B, 2*width, N)
        # dhdt_h, dhdt_a = torch.split(dhdt, self.width, dim=1)

        # return dhdt_h
        return x
    def a1(self,x):
        # print(x.shape)# torch.Size([121, 10, 16, 1656])
        x = x.permute(0, 1, 3, 2)
        x = self.fc1(x) 
        x = torch.relu(x)
        x = self.fc2(x) 
        
        x = x.permute(1, 2, 0, 3)
        # k1: torch.Size([10, 1656, 121, 3])
        return x
    def a2(self,ha):
        # ha = torch.cat([h, a], dim=2)   # (B, 10, 2*width, N)
        # ha = self.ha_conv(ha)           # (B, 10, width, N)
        h = F.gelu(ha)
        # print(x.shape, ref_score.shape, ref_x.shape, grid.shape)
        # x = torch.cat((x, ref_score, ref_x, grid), dim=-1)

        # x = self.fc0(ha)
        # 调整shape再送入 fc0
        ha = ha.permute(0, 2, 1).contiguous()  # [B, N, width]
        x = self.fc4(ha)                       # [B, N, width]
        # x = x.permute(0, 2, 1).contiguous()    # [B, width, N] 恢复原形状

        # print("x:",x.shape)#[10,121,16]
        x = self.Fmapping_low(x)

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
        # print("x:",x.shape)
        x = self.Extend(x, self.modes1)

        '''
        Project the constructed frequency weight back to original domain
        '''
        x1 = x.permute(3, 1, 0, 2)
        x1 = self.iFmapping(x1, self.Nt)
        x1 = self.iLmapping(x1, self.LBO_MATRIX)
        
        '''
        Parameterize on Spatiotemporal domain to increase the expressiveness of the model
        '''
        x = x1.permute(0,1,3,2)
        x = self.convt(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1, 3)
        # print("x:",x.shape)#x: torch.Size([121, 10, 16, 1656])
        return x
    def forward(self, x):
        x_coord, ref_x, ref_y, ref_score = x['x'], x['ref_x'], x['ref_y'], x['ref_score']

        ref_score = ref_score.reshape(-1, 1, 1) * torch.ones_like(x_coord)

        grid = self.get_grid(x_coord)
        # print("ref_score:",ref_score.shape,",ref_y:",ref_y.shape,",grid:",grid.shape)
        # x_in = torch.cat([ref_score, ref_y, grid], dim=-1)   # (B, N, 3)
        # 假设你只想取第0个索引的1656维
        ref_y_flat = ref_y[:, -1, :, :]  # [10, 121, 3]
        x_in = torch.cat([ref_score, ref_y_flat, grid], dim=-1)  # [10, 121, 10] (6+3+1)

        x_feat = self.fc0(x_in)   # (B, N, width)

        a_input = torch.cat([ref_x, x_coord], dim=-1)   # (B, N, 2)
        a_feat = self.fc3(a_input)                     # (B, N, width)

        depth_coord = (x_coord - ref_x) / float(self.steps)   # (B, N, coord_dim)
        depth_feat = self.coord_proj(depth_coord)             # (B, N, width)
        depth_scale = depth_feat.permute(0, 2, 1).contiguous()  # (B, width, N)
        
        h = x_feat.permute(0, 2, 1).contiguous()    # (10, 16, 121)
        a = a_feat.permute(0, 2, 1).contiguous()    # (10, 16, 121)
        a = self.a2(a) # [121, 10, 16, 1656]
        h = self.a2(h) # [121, 10, 16, 1656]
        depth_scale = self.a2(depth_scale) # [121, 10, 16, 1656]
        for _ in range(self.steps):
            # u1 = self.a2(a,h)#[121, 10, 16, 1656]
            k1 = self.func(a, h)#[121, 10, 16, 1656]
            # k1 = self.a1(k1)
            # print("k1:",k1.shape,",depth_scale:",depth_scale.shape,"h:",h.shape,",a:",a.shape)
            # exit()
            # k1: torch.Size([10, 1656, 121, 3]) ,depth_scale: torch.Size([10, 16, 121]) h: torch.Size([10, 16, 121]) ,a: torch.Size([10, 16, 121])
            k2 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k1)
            k3 = self.func(a + 0.5 * depth_scale, h + 0.5 * depth_scale * k2)
            k4 = self.func(a + depth_scale, h + depth_scale * k3)
            h = h + (depth_scale / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        x_out = self.a1(h)
        return x_out + ref_y.reshape(x_out.shape) 
        # return x_out

    def get_grid(self, x):
        B, N, C = x.shape     # batch, nodes, channels
        device = x.device
        grid = torch.linspace(0, 1, N, device=device)       # [N]
        grid = grid.unsqueeze(0).expand(B, N)               # [B, N]
        grid = grid.unsqueeze(-1)                           # [B, N, 1]        
        return grid
 
    def Lmapping(self, x, LBO_MATRIX, LBO_INVERSE):

        x = x = x.permute(0,1,3,2) 
        x = self.LBO_INVERSE @ x  
        x = x.permute(0,1,3,2) 
        return x
        
               
    def Fmapping(self, x, modes2): 

        x = x.permute(1,2,3,0)    
        x_ft = torch.fft.rfft(x)  
        x_ft = x_ft.permute(0,3,1,2) 
        return x_ft
    
    def iFmapping(self, x, Nt):

        x = x.permute(2,3,0,1) 
        
        x_rft = torch.fft.irfft(x, Nt)  
        
        return x_rft
        
    def iLmapping(self, x, LBO_MATRIX): 

        x = x.permute(3,0,1,2) 
        
        x = x @ LBO_MATRIX.T 
        
        return x
    
    def Fmapping_low(self, x): 

        x = x.permute(0,2,1)  
        
        x_ft = torch.fft.rfft(x) 
        
        x_ft = x_ft.permute(0,2,1) 
        
        return x_ft
    
    def Extend(self, x, modes): 

        scale = (1 / (x.shape[2] * modes))
        weights1 = nn.Parameter(scale*torch.rand(x.shape[1], x.shape[2], modes, dtype=torch.float)).cuda()
        x = torch.einsum("txbi,xio->txbo", x, weights1)
        
        return x
