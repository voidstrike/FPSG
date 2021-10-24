import torch
import pymesh

import numpy as np
import torch.nn as nn

from torch.autograd import Variable

from neuralnet_pytorch.metrics import emd_loss

'''Utility Classes'''
def emd_wrapper(pc1, pc2):
    return emd_loss(pc1, pc2, reduce='sum', sinkhorn=True)

class Template(object):
    def get_random_points(self):
        print('Need to be implemented')

    def get_regular_points(self):
        print('Need to be implemented')

class ShpereTemplate(Template):
    def __init__(self, device=0, grain=6):
        self.device = device
        self.dim = 3
        self.npoints = 0

    def get_random_points(self, shape, device='cuda'):
        assert shape[1] == 3, f'3 is expected in dimension 1, while got {shape}'
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.normal_(0, 1)
        rand_grid - rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True))
        return Variable(rand_grid)

    def get_regular_points(self, npoints=None, device='cuda'):
        if self.npoints != npoints:
            self.mesh = pymesh.generate_icosphere(1, [0, 0, 0], 4)
            self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0, 1).contiguous().unsqueeze(0)
            self.npoints = npoints
        
        return Variable(self.vertex.to(device))

class SquareTemplate(Template):
    def __init__(self, device=0):
        self.device = device
        self.dim = 2
        self.npoints = 0

    def get_random_points(self, shape, device='cuda'):
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.uniform_(0, 1)
        return Variable(rand_grid)

    def get_regular_points(self, npoints=2048, device='cuda'):
        if self.npoints != npoints:
            vertices, faces = self.generate_square(np.sqrt(npoints))
            self.mesh = pymesh.form_mesh(vertices=vertices, faces=faces)
            self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0, 1).contiguous().unsqueeze(0)
            self.npoints = npoints
        
        return Variable(self.vertex[:, :2].contiguous().to(device))

    @staticmethod
    def generate_square(grain):
        grain = int(grain) - 1

        faces, vertices = list(), list()

        for i in range(grain + 1):
            for j in range(grain + 1):
                vertices.append([i / grain, j / grain, 0])

        for i in range(grain + 1):
            for j in range(grain):
                faces.append([j + (grain + 1) * i, j + (grain + 1) * i + 1, j + (grain + 1) * (i - 1)])

        for i in range(grain):
            for j in range(1, grain+1):
                faces.append([j + (grain + 1) * i, j + (grain + 1) * i - 1, j + (grain + 1) * (i + 1)])

        return np.array(vertices), np.array(faces)


'''Utility Functions'''

def get_template(template_type, device=0):
    getter = {
        'SQUARE': SquareTemplate,
        'SPHERE': ShpereTemplate,
    }
    template = getter.get(template_type, 'Invalid template')
    return template(device=device)

def euclidean_dist(x, y):
    # Copy from prototypical network
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    assert d == y.size(1), 'Inconsistent dimension between tensor X and Y'

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def build_pc_proto(ttl_class, pred_label, pc_proto_mat):
    # Create pc prototype matrix via index
    # ans = torch.zeros(len(pred_label), ttl_class).scatter_(1, pred_label.unsqueeze(1), 1.)
    # print(ttl_class)
    # print(pred_label.shape)
    # print(pc_proto_mat.shape)
    ans = torch.zeros(len(pred_label), ttl_class).cuda().scatter_(1, pred_label, 1.)
    return torch.mm(ans, pc_proto_mat)

