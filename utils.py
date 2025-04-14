import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn
import numpy as np
import os
from PIL import Image

def concat_elu(x):
    axis = len(x.size()) - 3
    return F.elu(torch.cat([x, -x], dim=axis))

def log_sum_exp(x):
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def log_prob_from_logits(x):
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))

def discretized_mix_logistic_loss(x, l):
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = list(x.size())
    ls = list(l.size())
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2*nr_mix], min=-7.)
    coeffs = F.tanh(l[:, :, :, :, 2*nr_mix:3*nr_mix])
    x = x.contiguous()
    x = x.unsqueeze(-1) + Variable(torch.zeros(xs + [nr_mix]).to(x.device), requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)
    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*F.softplus(mid_in)
    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1.-inner_inner_cond)*(log_pdf_mid - np.log(127.5))
    inner_cond = (x > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1.-inner_cond)*inner_inner_out
    cond = (x < -0.999).float()
    log_probs = cond * log_cdf_plus + (1.-cond)*inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
    return -torch.sum(log_sum_exp(log_probs))

def to_one_hot(tensor, n, fill_with=1.):
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return Variable(one_hot)

def sample_from_discretized_mix_logistic(l, nr_mix):
    l = l.permute(0,2,3,1)
    ls = list(l.size())
    xs = ls[:-1] + [3]
    logit_probs = l[:,:,:, :nr_mix]
    l = l[:,:,:, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    temp = torch.FloatTensor(logit_probs.size())
    if l.is_cuda: temp = temp.cuda()
    temp.uniform_(1e-5, 1.-1e-5)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=3)
    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    means = torch.sum(l[:,:,:,:, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(l[:,:,:,:, nr_mix:2*nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(F.tanh(l[:,:,:,:, 2*nr_mix:3*nr_mix]) * sel, dim=4)
    u = torch.FloatTensor(means.size())
    if l.is_cuda: u = u.cuda()
    u.uniform_(1e-5, 1.-1e-5)
    u = Variable(u)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.-u))
    x0 = torch.clamp(x[:,:,:,0], min=-1., max=1.)
    x1 = torch.clamp(x[:,:,:,1] + coeffs[:,:,:,0]*x0, min=-1., max=1.)
    x2 = torch.clamp(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, min=-1., max=1.)
    out = torch.cat([x0.unsqueeze(3), x1.unsqueeze(3), x2.unsqueeze(3)], dim=3)
    out = out.permute(0,3,1,2)
    return out

def sample(model, sample_batch_size, obs, sample_op, labels=None):
    model.train(False)
    with torch.no_grad():
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.to(next(model.parameters()).device)
        if labels is None and hasattr(model, 'class_embedding'):
            labels = torch.zeros(sample_batch_size, dtype=torch.long, device=data.device)
        for i in range(obs[1]):
            for j in range(obs[2]):
                data_v = data
                if hasattr(model, 'class_embedding'):
                    out = model(data_v, labels, sample=True)
                else:
                    out = model(data_v, sample=True)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

class mean_tracker:
    def __init__(self):
        self.sum = 0
        self.count = 0
    def update(self, new_value):
        self.sum += new_value
        self.count += 1
    def get_mean(self):
        return self.sum / self.count
    def reset(self):
        self.sum = 0
        self.count = 0
         
class ratio_tracker:
    def __init__(self):
        self.sum = 0
        self.count = 0
    def update(self, new_value, new_count):
        self.sum += new_value
        self.count += new_count
    def get_ratio(self):
        return self.sum / self.count
    def reset(self):
        self.sum = 0
        self.count = 0
        
def check_dir_and_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        
def save_images(tensor, images_folder_path, label=''):
    os.makedirs(images_folder_path, exist_ok=True)
    for i, img_tensor in enumerate(tensor):
        img = Image.fromarray((img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), mode='RGB')
        img_path = f"{images_folder_path}/{label}_image_{i+1:02d}.png"
        img.save(img_path)
