import os
import torch

import torchvision.transforms as tfs
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

from .utils import extract_episode

# PLY file handler
def ply_reader(file_path):
    n_verts = 2048
    with open(file_path, 'r') as f:
        while True:
            cur = f.readline().strip()
            if cur == 'end_header':
                break

            cur = cur.split(' ')
            if len(cur) > 2 and cur[1] == 'vertex':
                n_verts = min(int(cur[2]), n_verts)

        vertices = [[float(s) for s in f.readline().strip().split(' ')] for _ in range(n_verts)]

    return vertices

class FewShotSubModelNet(Dataset):
    def __init__(self, config_path, loader=ply_reader, transform=None, tgt_transform=None, data_argument=False, n_pts=2048):
        super(FewShotSubModelNet, self).__init__()
        self.imgs = list()
        self.pcs = list()
        self.loader = loader

        with open(config_path, 'r') as f:
            for eachLine in f.readlines():
                tmp_cmp = eachLine.rstrip('\n').split('\t')
                self.imgs.append(tmp_cmp[0])
                self.pcs.append(tmp_cmp[1])

        self.tfs = transform
        self.tgt_tfs = tgt_transform
        self.data_argument = data_argument
        self.n_pts = n_pts

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pc_path = self.pcs[index]

        img = Image.open(img_path).convert('RGB')
        if self.tfs is not None:
            img = self.tfs(img)

        sample = self.loader(pc_path)
        point_set = np.asarray(sample, dtype=np.float32)

        if point_set.shape[0] < self.n_pts:
            choice = np.random.choice(len(point_set), self.n_pts - point_set.shape[0], replace=True)
            aux_pc = point_set[choice, :]
            point_set = np.concatenate((point_set, aux_pc))

        center_point = np.expand_dims(np.mean(point_set, axis=0), 0)
        point_set = point_set - center_point
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist

        if self.data_argument:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set).contiguous()
        
        return img, point_set

    def __len__(self, ):
        return len(self.imgs)


class FewShotModelNet(Dataset):
    def __init__(self, config_path, auxiliary_dir, n_classes, n_support, n_query, transform=None, tgt_transform=None):
        super(FewShotModelNet, self).__init__()

        # Store the img_path & ply path for every data point
        self.data_corpus = list()
        with open(config_path, 'r') as f:
            for eachItem in f.readlines():
                self.data_corpus.append(eachItem.rstrip('\n'))

        self.tfs, self.tgt_tfs = transform, tgt_transform

        # Store the img_path & ply path for a specific class -- faster access
        self.reference = dict()
        self.auxiliary_dir = auxiliary_dir
        self._build_reference()
        

        self.n_way = n_classes
        self.n_support = n_support
        self.n_query = n_query

        self.seen_index = [False] * len(self.data_corpus)
        self.cache = dict()

    def __getitem__(self, index):
        img_path, pc_path = self.data_corpus[index].split('\t')
        data_instance_class = img_path.split('/')[7] # Hard code ?

        query_matrix = {
            'class': data_instance_class,
            'img_data': self.reference[data_instance_class]['imgs'],
            'pc_data': self.reference[data_instance_class]['pcs'],
        }

        return extract_episode(self.n_support, self.n_query, query_matrix)

        # index = index.item()
        # if not self.seen_index[index]:
        #     img_path, pc_path = self.data_corpus[index].split('\t')
        #     data_instance_class = img_path.split('/')[7] # Hard code ?

        #     query_matrix = {
        #         'class': data_instance_class,
        #         'img_data': self.reference[data_instance_class]['imgs'],
        #         'pc_data': self.reference[data_instance_class]['pcs'],
        #     }

        #     self.cache[index] = extract_episode(self.n_support, self.n_query, query_matrix)
        #     self.seen_index[index] = True
        # return self.cache[index]

    def _build_reference(self):
        assert self.auxiliary_dir is not None, 'Auxiliary folder is not available!!!'

        for eachFile in os.listdir(self.auxiliary_dir):
            if not eachFile.endswith('.txt'):
                continue

            class_name = eachFile.split('.')[0].split('+')[1]
            self.reference[class_name] = dict()

            class_ds = FewShotSubModelNet(os.path.join(self.auxiliary_dir, eachFile), transform=self.tfs, tgt_transform=self.tgt_tfs)
            loader = DataLoader(class_ds, batch_size=len(class_ds), shuffle=False)

            for stacked_img, stacked_pc in loader:
                self.reference[class_name]['imgs'] = stacked_img
                self.reference[class_name]['pcs'] = stacked_pc
                break # Follow the protonet, only need one sample because batch_size equal to the dataset length

        
    def __len__(self, ):
        return len(self.data_corpus)
