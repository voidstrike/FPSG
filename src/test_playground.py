import torch
import argparse

import torch.nn as nn
import numpy as np
import torchvision.transforms as tfs

from datasets.modelnet import FewShotSubModelNet, FewShotModelNet
from datasets.utils import EpisodicBatchSampler
from torch.utils.data import DataLoader

from pointnet.model import PointNetfeat
from models.few_shot import ImgPCProtoNet
from models.image_net import ImageEncoderWarpper
from models.point_cloud_net import PCEncoder, PCDecoder
from metrics.evaluation_metrics import emd_approx

_modelnet_tfs = tfs.Compose([
    tfs.CenterCrop(550),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])


def main(opt):
    t1 = torch.ones(4, 100, 3).cuda()
    t2 = torch.ones(4, 100, 3).cuda()

    tout = emd_approx(t1, t2)
    print(tout.shape)
    t3 = tout.sum()
    print(t3.shape)
    # config_file = './modelnet_train.txt'
    # auxiliary_dir = './extra_files/'
    # n_way = 30
    # n_episode = 100

    # test_ds = FewShotModelNet(config_file, auxiliary_dir, n_classes=n_way, n_support=5, n_query=5, transform=_modelnet_tfs)
    # sampler = EpisodicBatchSampler(len(test_ds), n_way, n_episode)
    
    # dl = DataLoader(test_ds, batch_sampler=sampler, num_workers=0)

    # for idx, corpus in enumerate(dl):
    #     print(idx)
        # print(f'Support Image Set has Shape: { corpus["xs"].shape }')
        # print(f'Query Image Set has Shape: {corpus["xq"].shape}')
        # print(f'Support Point CLoud Set has Shape: {corpus["pcs"].shape}')
        # print(f'Query Point CLoud Set has Shape: {corpus["pcq"].shape}')
        # break

    # img_support = torch.ones((5, 1, 3, 224, 224)).cuda()
    # img_query = torch.ones((5, 1, 3, 224, 224)).cuda()
    # pc_support = torch.ones((5, 1, 2048, 3)).cuda()
    # pc_query = torch.ones((5, 1, 2048, 3)).cuda()

    # tin = {
    #     'class': 'test',
    #     'xs': img_support,
    #     'xq': img_query,
    #     'pcs': pc_support,
    #     'pcq': pc_query,
    # }
    
    # img_encoder = ImageEncoderWarpper()
    # pc_encoder = PCEncoder()
    # pc_decoder = PCDecoder(configuration=opt)

    # model = ImgPCProtoNet(img_encoder, pc_encoder, pc_decoder)
    # model = model.cuda()
    # tout = model.loss(tin)

    # print(tout)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_slaves', type=int, default=1),
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--bottleneck_size', type=int, default=1536)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--template_type', type=str, default="SQUARE")
    parser.add_argument('--hidden_neurons', type=int, default=512)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dim_template', type=int, default=2)

    conf = parser.parse_args()

    main(conf)
