import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .utils import get_template
from pointnet.model import PointNetfeat

class PCEncoder(nn.Module):
    def __init__(self, core='pointnet'):
        super(PCEncoder, self).__init__()

        self.pc_encoder = None
        if core == 'pointnet':
            self.pc_encoder = PointNetfeat()
        elif core == 'dgcnn':
            #TODO
            pass
        else:
            raise NotImplementedError(f'Unsupported Point Cloud Encoder Core: {core}')
    
    def forward(self, x):
        pc_feat, _, _ = self.pc_encoder(x)
        return pc_feat

class PCDecoder(nn.Module):
    # Adopt Altas-Net based Decoder
    def __init__(self, configuration, num_prototypes=1, num_pts=2048):
        super(PCDecoder, self).__init__()

        self.opt = configuration
        self.device = configuration.device
        self.num_slaves = configuration.num_slaves
        self.num_prototypes = num_prototypes

        self.num_pts_per_cluster = num_pts // self.num_prototypes

        self.cluster_pool = nn.ModuleList([MLPCluster(self.opt, self.num_pts_per_cluster, self.num_slaves) for _ in range(self.num_prototypes)])

    def forward(self, img_feat, pc_feat):
        # The shape of img_feat should be Batch * 512
        # The shape of pc_feat should be Batch * Num_proto * 1024
        pc_feat = pc_feat.transpose_(0, 1)
        output_points = torch.cat([self.cluster_pool[idx](img_feat, pc_feat[idx]) for idx in range(self.num_prototypes)], dim=2)
        return output_points.transpose(1, 2).contiguous()

class MLPCluster(nn.Module):
    def __init__(self, opt, ttl_point, num_primitives):
        super(MLPCluster, self).__init__()
        self.opt = opt
        
        self.num_slaves = num_primitives
        self.num_per_slave = ttl_point // num_primitives
        self.template = [get_template(opt.template_type, device=opt.device) for _ in range(num_primitives)]
        print(f'Initialized new MLP cluster w/ {self.num_slaves} slave(s)')

        self.slave_pool = nn.ModuleList([MLPSlave(opt) for _ in range(num_primitives)])

    def forward(self, x, proto_mat):
        # Generate input random grid for each slave-mlp
        input_points = [
            self.template[idx].get_random_points(torch.Size((1, self.template[idx].dim, self.num_per_slave))) 
            for idx in range(self.num_slaves)
            ]

        # Get the real input
        x = torch.cat([x, proto_mat], dim=1)

        tmp_factor = 1 if self.num_slaves == 1 else 3

        output_points = torch.cat([self.slave_pool[idx](input_points[idx], x.unsqueeze(2)).unsqueeze(1) for idx in range(self.num_slaves)],\
             dim=tmp_factor).squeeze(1)

        return output_points

class MLPSlave(nn.Module):
    # Simple vanilla shared-MLP
    def __init__(self, opt):
        super(MLPSlave, self).__init__()
        self.opt = opt
        self.bottleneck_size = opt.bottleneck_size
        self.input_size = opt.dim_template
        self.dim_output = 3
        self.hidden_neurons = opt.hidden_neurons
        self.num_layers = opt.num_layers

        print(f'Initialized new MLP slave w/ hidden size equals {self.hidden_neurons}, num_layers equals {self.num_layers}, and activation function {opt.activation}')

        self.conv1 = nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)
        self.conv_list = nn.ModuleList([nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for _ in range(self.num_layers)])
        self.last_conv = nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        self.bn1 = nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_neurons)
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(self.hidden_neurons) for _ in range(self.num_layers)])
        
        self.activation = get_activation(opt.activation)

    def forward(self, x, latent):
        x = self.conv1(x) + latent
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))

        return torch.tanh(self.last_conv(x))

class AuxClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_layer=3):
        super(AuxClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)

        self.dropout = nn.Dropout(p=.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self._init_weight()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        # return x # logit

    def _init_weight(self,):
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)



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