import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from torch.autograd import Variable
import os

class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=0)
                                       for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                                     resnet_nonlinearity, skip_connection=1)
                                        for _ in range(nr_resnet)])
    def forward(self, u, ul):
        u_list, ul_list = [], []
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list.append(u)
            ul_list.append(ul)
        return u_list, ul_list

class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=1)
                                       for _ in range(nr_resnet)])
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                                     resnet_nonlinearity, skip_connection=2)
                                        for _ in range(nr_resnet)])
    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))
        return u, ul

class ConditionalPixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, cond_dim):
        super(ConditionalPixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([
            gated_resnet(nr_filters, down_shifted_conv2d, resnet_nonlinearity,
                         skip_connection=0, use_condition=True, cond_dim=cond_dim)
            for _ in range(nr_resnet)
        ])
        self.ul_stream = nn.ModuleList([
            gated_resnet(nr_filters, down_right_shifted_conv2d, resnet_nonlinearity,
                         skip_connection=1, use_condition=True, cond_dim=cond_dim)
            for _ in range(nr_resnet)
        ])
    def forward(self, u, ul, cond):
        u_list, ul_list = [], []
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, cond=cond)
            ul = self.ul_stream[i](ul, a=u, cond=cond)
            u_list.append(u)
            ul_list.append(ul)
        return u_list, ul_list

class ConditionalPixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity, cond_dim):
        super(ConditionalPixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        self.u_stream = nn.ModuleList([
            gated_resnet(nr_filters, down_shifted_conv2d, resnet_nonlinearity,
                         skip_connection=1, use_condition=True, cond_dim=cond_dim)
            for _ in range(nr_resnet)
        ])
        self.ul_stream = nn.ModuleList([
            gated_resnet(nr_filters, down_right_shifted_conv2d, resnet_nonlinearity,
                         skip_connection=2, use_condition=True, cond_dim=cond_dim)
            for _ in range(nr_resnet)
        ])
    def forward(self, u, ul, u_list, ul_list, cond):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop(), cond=cond)
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1), cond=cond)
        return u, ul

class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('Currently only concat_elu is supported.')
        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))
        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([
            PixelCNNLayer_down(down_nr_resnet[i], nr_filters, self.resnet_nonlinearity)
            for i in range(3)
        ])
        self.up_layers = nn.ModuleList([
            PixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity)
            for _ in range(3)
        ])
        self.downsize_u_stream = nn.ModuleList([
            down_shifted_conv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.downsize_ul_stream = nn.ModuleList([
            down_right_shifted_conv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.upsize_u_stream = nn.ModuleList([
            down_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.upsize_ul_stream = nn.ModuleList([
            down_right_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3), shift_output_down=True)
        self.ul_init = nn.ModuleList([
            down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(1,3), shift_output_down=True),
            down_right_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,1), shift_output_right=True)
        ])
        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, sample=False):
        if self.init_padding is not sample:
            xs = list(x.size())
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding
        if sample:
            xs = list(x.size())
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out
            if i != 2:
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]
        u = u_list.pop()
        ul = ul_list.pop()
        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)
        x_out = self.nin_out(F.elu(ul))
        assert len(u_list) == len(ul_list) == 0, "Mismatch in skip connection buffers"
        return x_out

class ConditionalPixelCNN(nn.Module):
    def __init__(self, num_classes, embedding_dim=64, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                 resnet_nonlinearity='concat_elu', input_channels=3):
        super(ConditionalPixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('Currently only concat_elu is supported.')
        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        self.cond_project_u = nn.Linear(embedding_dim, nr_filters)
        self.cond_project_ul = nn.Linear(embedding_dim, nr_filters)

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([
            ConditionalPixelCNNLayer_down(down_nr_resnet[i], nr_filters, self.resnet_nonlinearity, cond_dim=embedding_dim)
            for i in range(3)
        ])
        self.up_layers = nn.ModuleList([
            ConditionalPixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity, cond_dim=embedding_dim)
            for _ in range(3)
        ])
        self.downsize_u_stream = nn.ModuleList([
            down_shifted_conv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.downsize_ul_stream = nn.ModuleList([
            down_right_shifted_conv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.upsize_u_stream = nn.ModuleList([
            down_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.upsize_ul_stream = nn.ModuleList([
            down_right_shifted_deconv2d(nr_filters, nr_filters, stride=(2,2))
            for _ in range(2)
        ])
        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,3), shift_output_down=True)
        self.ul_init = nn.ModuleList([
            down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(1,3), shift_output_down=True),
            down_right_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2,1), shift_output_right=True)
        ])
        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, labels, sample=False):
        # Obtain the class embeddings and project for early fusion
        labels = torch.where(labels < 0, torch.zeros_like(labels), labels)
        emb = self.class_embedding(labels)  # Now safe to use
        cond_u = self.cond_project_u(emb)                 
        cond_ul = self.cond_project_ul(emb)              
        cond_u = cond_u.unsqueeze(-1).unsqueeze(-1)       
        cond_ul = cond_ul.unsqueeze(-1).unsqueeze(-1)        # (B, nr_filters, 1, 1)
        
        xs = list(x.size())
        padding = torch.ones(xs[0], 1, xs[2], xs[3], device=x.device, dtype=x.dtype)
        x = torch.cat((x, padding), 1)
        
        u = self.u_init(x)  # (B, nr_filters, H, W)
        ul = self.ul_init[0](x) + self.ul_init[1](x)
        # Early fusion: add condition projections
        u = u + cond_u
        ul = ul + cond_ul
        
        u_list = [u]
        ul_list = [ul]
        for i in range(3):
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1], cond=emb)
            u_list += u_out
            ul_list += ul_out
            if i != 2:
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]
        u = u_list.pop()
        ul = ul_list.pop()
        for i in range(3):
            u, ul = self.down_layers[i](u, ul, u_list, ul_list, cond=emb)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)
        x_out = self.nin_out(F.elu(ul))
        assert len(u_list) == len(ul_list) == 0, "Mismatch in skip connection buffers"
        return x_out

class random_classifier(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(random_classifier, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        self.fc = nn.Linear(3, NUM_CLASSES)
        print("Random classifier initialized")
        if os.path.join(os.path.dirname(__file__), 'models') not in os.listdir():
            os.mkdir(os.path.join(os.path.dirname(__file__), 'models'))
        torch.save(self.state_dict(), os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth'))
    def forward(self, x, device):
        return torch.randint(0, self.NUM_CLASSES, (x.shape[0],)).to(device)
