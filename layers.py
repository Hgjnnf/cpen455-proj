import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
from utils import concat_elu

class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        x = x.permute(0, 2, 3, 1)
        shp = list(x.size())
        out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)

def down_shift(x, pad=None):
    shp = list(x.size())
    x = x[:, :, :shp[2]-1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)

def right_shift(x, pad=None):
    shp = list(x.size())
    x = x[:, :, :, :shp[3]-1]
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)

class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1),
                 shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()
        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad = nn.ZeroPad2d(((filter_size[1]-1)//2, (filter_size[1]-1)//2, filter_size[0]-1, 0))
        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)
        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0,0,1,0)))
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.norm == 'batch_norm':
            x = self.bn(x)
        return self.down_shift(x) if self.shift_output_down else x

class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride
    def forward(self, x):
        x = self.deconv(x)
        shp = list(x.size())
        return x[:, :, :shp[2]-self.filter_size[0]+1, (self.filter_size[1]-1)//2 : shp[3] - (self.filter_size[1]-1)//2]

class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                 shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()
        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1]-1, 0, filter_size[0]-1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm
        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)
        if shift_output_right:
            self.right_shift = lambda x: right_shift(x, pad=nn.ZeroPad2d((1,0,0,0)))
    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        if self.norm == 'batch_norm':
            x = self.bn(x)
        return self.right_shift(x) if self.shift_output_right else x

class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                 shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                            stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride
    def forward(self, x):
        x = self.deconv(x)
        shp = list(x.size())
        return x[:, :, :shp[2]-self.filter_size[0]+1, :shp[3]-self.filter_size[1]+1]

# Modified gated_resnet for conditional fusion
class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0,
                 use_condition=False, cond_dim=None):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.use_condition = use_condition
        self.conv_input = conv_op(2 * num_filters, num_filters)
        if skip_connection != 0:
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)
        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)
        if self.use_condition:
            if cond_dim is None:
                raise ValueError("cond_dim must be specified if use_condition is True")
            self.cond_proj = nn.Linear(cond_dim, num_filters)
    def forward(self, og_x, a=None, cond=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None:
            x += self.nin_skip(self.nonlinearity(a))
        if self.use_condition and cond is not None:
            cond_projection = self.cond_proj(cond)
            cond_projection = cond_projection.unsqueeze(-1).unsqueeze(-1)
            x = x + cond_projection
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * torch.sigmoid(b)
        return og_x + c3
