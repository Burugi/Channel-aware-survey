import torch
import torch.nn as nn
import math

class Splitting(nn.Module):
    """Signal splitting module"""
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, ::2, :]

    def odd(self, x):
        return x[:, 1::2, :]

    def forward(self, x):
        return self.even(x), self.odd(x)

class Interactor(nn.Module):
    """Interaction module"""
    def __init__(self, in_planes, kernel=5, dropout=0.5, groups=1, hidden_size=1, INN=True):
        super(Interactor, self).__init__()
        self.modified = INN
        self.kernel_size = kernel
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.groups = groups
        self.split = Splitting()
        
        # Calculate padding
        pad_l = pad_r = (kernel - 1) // 2 + 1 
        
        # Build modules
        def build_conv_block(in_c, out_c):
            return nn.Sequential(
                nn.ReplicationPad1d((pad_l, pad_r)),
                nn.Conv1d(in_c, out_c, kernel, groups=groups),
                nn.LeakyReLU(0.01),
                nn.Dropout(dropout),
                nn.Conv1d(out_c, in_c, 3, groups=groups),
                nn.Tanh()
            )
        
        self.phi = build_conv_block(in_planes, int(in_planes * hidden_size))
        self.psi = build_conv_block(in_planes, int(in_planes * hidden_size))
        self.P = build_conv_block(in_planes, int(in_planes * hidden_size))
        self.U = build_conv_block(in_planes, int(in_planes * hidden_size))

    def forward(self, x):
        # Split signal
        x_even, x_odd = self.split(x)
        
        if self.modified:
            # [Batch, Time, Channel] -> [Batch, Channel, Time]
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            
            # Apply transformations
            d = x_odd.mul(torch.exp(self.phi(x_even)))
            c = x_even.mul(torch.exp(self.psi(x_odd)))
            x_even_update = c + self.U(d)
            x_odd_update = d - self.P(c)
            
            # [Batch, Channel, Time] -> [Batch, Time, Channel]
            x_even_update = x_even_update.permute(0, 2, 1)
            x_odd_update = x_odd_update.permute(0, 2, 1)
        else:
            x_even = x_even.permute(0, 2, 1)
            x_odd = x_odd.permute(0, 2, 1)
            
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            
            x_even_update = c.permute(0, 2, 1)
            x_odd_update = d.permute(0, 2, 1)

        return x_even_update, x_odd_update

class EncoderTree(nn.Module):
    """Encoder tree module"""
    def __init__(self, in_planes, num_levels, kernel_size, dropout, groups, hidden_size, INN):
        super(EncoderTree, self).__init__()
        self.levels = num_levels
        self.interactors = nn.ModuleList()
        
        for i in range(num_levels):
            self.interactors.append(
                Interactor(in_planes=in_planes, kernel=kernel_size, dropout=dropout,
                          groups=groups, hidden_size=hidden_size, INN=INN)
            )
            
    def zip_up_the_pants(self, even, odd):
        """Interleave even and odd sequences"""
        even = even.permute(1, 0, 2)
        odd = odd.permute(1, 0, 2)
        
        mlen = min(even.shape[0], odd.shape[0])
        _list = []
        for i in range(mlen):
            _list.extend([even[i:i+1], odd[i:i+1]])
            
        if even.shape[0] > odd.shape[0]:
            _list.append(even[-1:])
            
        return torch.cat(_list, 0).permute(1, 0, 2)
    
    def forward(self, x, current_level=0):
        if current_level == self.levels:
            return x
            
        x_even_update, x_odd_update = self.interactors[current_level](x)
        
        x_even = self.forward(x_even_update, current_level + 1)
        x_odd = self.forward(x_odd_update, current_level + 1)
        
        return self.zip_up_the_pants(x_even, x_odd)