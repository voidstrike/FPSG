import os
import torch

import torchvision.transforms as tfs
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image

from .utils import extract_episode

_SHAPENET_ID2NAME = {
    '02691156': 'airplane',
    '02880940': 'bowl',
    '02942699': 'camera',
    '02958343': 'car',
    '02992529': 'cellphone',
    '03001627': 'chair',
    '03046257': 'clock',
    '03211117': 'monitor',
    '03325088': 'faucet',
    '03593526': 'jar',
    '03797390': 'mug',
    '04004475': 'printer',
    '04099429': 'rocket',
}

_SHAPENET_NAME2ID = {_SHAPENET_ID2NAME[eachKey]: eachKey for eachKey in _SHAPENET_ID2NAME.keys()}

class FewShotSubShapeNet(Dataset):
    def __init__(self, config_path, transform=None, tgt_transform=None, data_argument=False, n_pts=2048):
        super(FewShotSubShapeNet, self).__init__()
        self.imgs = list()
        self.pcs = list()

        with open(config_path, 'r') as f:
            for eachLine in f.readlines():
                item_path = filename = eachLine.rstrip('\n')
                npy_file = os.path.join(item_path, 'npy_file.npy')
                view_root = os.path.join(item_path, 'images')

                if not os.path.exists(npy_file):
                    continue

                views = list()
                for view in os.listdir(view_root):
                    views.append(os.path.join(view_root, view))

                self.pcs.append(npy_file)
                self.imgs.append(views)

        self.pc_data = list()
        for idx in range(len(self.pcs)):
            try:
                pc = np.load(self.pcs[idx])
            except:
                raise Exception('Unexpected Error!')

            choice = np.random.choice(15000, n_pts)
            pc = pc[choice, :]
            self.pc_data.append(pc)

        self.tfs = transform
        self.tgt_tfs = tgt_transform
        self.data_argument = data_argument
        self.n_pts = n_pts

    def __getitem__(self, index):
        img_path = self.imgs[index][0]
        pc = self.pc_data[index]

        img = Image.open(img_path).convert('RGB')
        if self.tfs is not None:
            img = self.tfs(img)

        point_set = np.asarray(pc, dtype=np.float32)

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


class FewShotShapeNet(Dataset):
    def __init__(self, config_path, auxiliary_dir, n_classes, n_support, n_query, transform=None, tgt_transform=None):
        super(FewShotShapeNet, self).__init__()

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
        item_path = self.data_corpus[index]
        data_instance_class = item_path.split('/')[5] # Hard code ?

        query_matrix = {
            'class': _SHAPENET_ID2NAME[data_instance_class],
            'img_data': self.reference[data_instance_class]['imgs'],
            'pc_data': self.reference[data_instance_class]['pcs'],
        }

        return extract_episode(self.n_support, self.n_query, query_matrix)

    def _build_reference(self):
        assert self.auxiliary_dir is not None, 'Auxiliary folder is not available!!!'

        for eachFile in os.listdir(self.auxiliary_dir):
            if not eachFile.endswith('.txt'):
                continue

            class_name = eachFile.split('.')[0].split('+')[1]
            self.reference[class_name] = dict()

            class_ds = FewShotSubShapeNet(os.path.join(self.auxiliary_dir, eachFile), transform=self.tfs, tgt_transform=self.tgt_tfs)
            print(f'{eachFile}: {len(class_ds)}')
            # loader = DataLoader(class_ds, batch_size=len(class_ds), shuffle=False)
            loader = DataLoader(class_ds, batch_size=min(200, len(class_ds)))

            for stacked_img, stacked_pc in loader:
                self.reference[class_name]['imgs'] = stacked_img
                self.reference[class_name]['pcs'] = stacked_pc
                break # Follow the protonet, only need one sample because batch_size equal to the dataset length

        
    def __len__(self, ):
        return len(self.data_corpus)