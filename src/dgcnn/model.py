import os
import sys
import copy
import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Borrowed from DGCNN: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (Batch_size, num_points, k)

    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn(x, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class DGCNNfeat(nn.Module):
    def __init__(self, embeding_dim=1024, num_neighbors=20, dual_pool=True):
        super(DGCNNfeat, self).__init__()

        self.dual_flag = dual_pool
        self.emb_dims = embeding_dim // 2 if dual_pool else embeding_dim
        self.k = num_neighbors

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(negative_slope=.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False), nn.BatchNorm1d(self.emb_dims), nn.LeakyReLU(negative_slope=.2))

    def forward(self, x):
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)

        if self.dual_flag:
            x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
            x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
            final_feat = torch.cat((x1, x2), 1)
        else:
            final_feat = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        return final_feat
    