import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .utils import get_template
from pointnet.model import PointNetfeat
from dgcnn.model import DGCNNfeat

# Wrapper for PointNetfeat
class PointNetWrapper(nn.Module):
    def __init__(self):
        super(PointNetWrapper, self).__init__()
        self.pointnet_feat_extractor = PointNetfeat()

    def forward(self, x):
        pc_feat, _, _ = self.pointnet_feat_extractor(x)
        return pc_feat

# A Point Cloud Encoder wrapper
class PCEncoder(nn.Module):
    def __init__(self, core='pointnet'):
        super(PCEncoder, self).__init__()

        self.pc_encoder = None
        if 'pointnet' == core:
            self.pc_encoder = PointNetWrapper()
        elif 'dgcnn' == core:
            self.pc_encoder = DGCNNfeat()
        else:
            raise NotImplementedError(f'Unsupported Point Cloud Encoder Core: {core}')

    def forward(self, x):
        assert self.pc_encoder is not None, 'Point Cloud Encoder is not initialized'
        return (self.pc_encoder(x))

class MLPDeformer(nn.Module):
    def __init__(self, conf):
        super(MLPDeformer, self).__init__()
        self.layer_size = 128
        self.input_size = conf.ori_dim
        self.dim_output = conf.raw_dim

        self.conv1 = nn.Conv1d(self.input_size, self.layer_size, 1)
        self.conv2 = nn.Conv1d(self.layer_size, self.layer_size, 1)
        self.conv3 = nn.Conv1d(self.layer_size, self.dim_output, 1)

        self.bn1 = nn.BatchNorm1d(self.layer_size)
        self.bn2 = nn.BatchNorm1d(self.layer_size)
        self.activation = get_activation(conf.activation)
    
    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        return torch.tanh(self.conv3(x))

class PrimitiveNode(nn.Module):
    """ Elementary Structure of the PC Decoder """
    def __init__(self, conf, input_dim):
        super(PrimitiveNode, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 3

        print(f'New Primitive Node initialized, hidden configuration {[self.input_dim, self.input_dim, self.input_dim // 2, self.input_dim // 4, self.output_dim]}')
        
        self.conv1 = nn.Conv1d(self.input_dim, self.input_dim, 1)
        self.conv2 = nn.Conv1d(self.input_dim, self.input_dim // 2, 1)
        self.conv3 = nn.Conv1d(self.input_dim // 2, self.input_dim // 4, 1)
        self.conv4 = nn.Conv1d(self.input_dim // 4, self.output_dim, 1)

        self.bn1 = nn.BatchNorm1d(self.input_dim)
        self.bn2 = nn.BatchNorm1d(self.input_dim // 2)
        self.bn3 = nn.BatchNorm1d(self.input_dim // 4)
        self.activation = get_activation(conf.activation)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        return torch.tanh(self.conv4(x))

class PrimitiveCluster(nn.Module):
    def __init__(self, conf, deformer, ttl_pts, nodes):
        super(PrimitiveCluster, self).__init__()
        self.conf = conf
        self.deformer = deformer
        self.num_nodes = nodes
        self.pts_per_node = ttl_pts // self.num_nodes
        self.template = [get_template(self.conf.template_type, device=self.conf.device) for _ in range(self.num_nodes)]

        print(f'New Primitive Cluster initialized, this cluster contains {self.num_nodes} node(s).')

        self.node_pool = nn.ModuleList([
            PrimitiveNode(conf, self.conf.raw_dim + self.conf.bottleneck_size) for _ in range(self.num_nodes)
        ])

    def forward(self, x):
        raw_pts = [
            self.template[idx].get_random_points(torch.Size((x.size(0), self.template[idx].dim, self.pts_per_node))) for idx in range(self.num_nodes)
        ]
        deformed_pts = [
            self.deformer(sample_pts) for sample_pts in raw_pts
        ]

        x = x.unsqueeze(2).repeat(1, 1, self.pts_per_node).contiguous()

        if self.num_nodes == 1:
            out_pts = torch.cat([self.node_pool[idx](torch.cat((x, deformed_pts[idx]), dim=1)).unsqueeze(1) for idx in range(self.num_nodes)], dim=1).squeeze(1)
        else:
            out_pts = torch.cat([self.node_pool[idx](torch.cat((x, deformed_pts[idx]), dim=1)).unsqueeze(1) for idx in range(self.num_nodes)], dim=3).squeeze(1)

        return out_pts

class PCDecoder(nn.Module):
    """ Modified AtlasNet Decoder with multiple patches and global shape priors"""
    def __init__(self, conf, num_pts=2048):
        super(PCDecoder, self).__init__()

        self.conf = conf
        self.device = self.conf.device
        self.num_nodes = self.conf.num_nodes
        self.num_clusters = self.conf.num_clusters
        self.num_pts_per_cluster = num_pts // self.num_clusters

        self.cluster_pool = nn.ModuleList([
            PrimitiveCluster(self.conf, MLPDeformer(self.conf), self.num_pts_per_cluster, self.num_nodes) for _ in range(self.num_clusters)
        ])

    def forward(self, hidden_feat):
        # The shape of hidden_feat should be Batch * (512 + 1024)
        output_points = torch.cat([self.cluster_pool[idx](hidden_feat) for idx in range(self.num_clusters)], dim=2)
        return output_points.transpose(1, 2).contiguous()


def get_activation(argument):
    getter = {
        'relu': F.relu,
        'sigmoid': F.sigmoid,
        'softplus': F.softplus,
        'logsigmoid': F.logsigmoid,
        'softsign': F.softsign,
        'tanh': torch.tanh,
    }

    return getter.get(argument, 'Invalid activation')