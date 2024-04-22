import numpy as np
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from bufferDataGet import *

class Buffer(nn.Module):
    def __init__(self, folder, buffer):
        super().__init__()
        self.buffer = buffer
        self.same_season = []
        self.other_season = []
        self.folder = folder

        self.arange_like = lambda x : torch.arange(x.size(0)).to(x.device)
        self.shuffle     = lambda x : x[torch.randperm(x.size(0))]

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])

    @property
    def t(self):
        return self.bt[:self.current_index]

    @property
    def valid(self):
        return self.is_valid[:self.current_index]

    def display(self, gen=None, epoch=-1):
        from torchvision.utils import save_image
        from PIL import Image

        if 'cifar' in self.args.dataset:
            shp = (-1, 3, 32, 32)
        else:
            shp = (-1, 1, 28, 28)

        if gen is not None:
            x = gen.decode(self.x)
        else:
            x = self.x

        save_image((x.reshape(shp) * 0.5 + 0.5), 'samples/buffer_%d.png' % epoch, nrow=int(self.current_index ** 0.5))
        #Image.open('buffer_%d.png' % epoch).show()
        print(self.y.sum(dim=0))

    def add_reservoir(self, x, y, logits, t):
        n_elem = x.size(0)
        save_logits = logits is not None

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)


            if save_logits:
                self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.bx.size(0), pdb.set_trace()
        assert idx_buffer.max() < self.by.size(0), pdb.set_trace()
        assert idx_buffer.max() < self.bt.size(0), pdb.set_trace()

        assert idx_new_data.max() < x.size(0), pdb.set_trace()
        assert idx_new_data.max() < y.size(0), pdb.set_trace()

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data]
        self.by[idx_buffer] = y[idx_new_data]
        self.bt[idx_buffer] = t

        if save_logits:
            self.logits[idx_buffer] = logits[idx_new_data]

    def shuffle_(self):
        indices = torch.randperm(self.current_index).to(self.args.device)
        self.bx = self.bx[indices]
        self.by = self.by[indices]


    def delete_up_to(self, remove_after_this_idx):
        self.bx = self.bx[:remove_after_this_idx]
        self.by = self.by[:remove_after_this_idx]

    def sample(self, amt):

        indices = np.random.choice(self.buffer, amt, replace=False) #抽样方法

        input_data, output_data = get_rehearsal_set(self.folder, indices)
        return input_data, output_data, indices

    def update_season(self, bi):
        i = bi - 365
        tmp_season = []
        while i >= 0:
            # print(i)
            tmp_season.extend([j for j in range(i, i + 60)])
            i -= 365
        self.same_season = list(set(self.buffer).intersection(tmp_season))
        self.other_season = list(set(self.buffer).difference(self.same_season))

    def season_sample(self, amt):
        #根据季节划分 buffer
        # print(self.same_season)
        # sindices = np.random.choice(self.same_season, amt, replace=False) #抽样方法
        oindices = np.random.choice(self.other_season, amt, replace=False)
        # print(sindices.dtype)
        # print(oindices.dtype)
        # indices = np.concatenate((oindices,sindices), axis=0) #抽样方法

        input_data, output_data = get_rehearsal_set(self.folder, oindices)
        return input_data, output_data, oindices

    def append_set(self, set):
        self.buffer.extend(set)

    def split(self, amt):
        indices = torch.randperm(self.current_index).to(self.args.device)
        return indices[:amt], indices[amt:]

