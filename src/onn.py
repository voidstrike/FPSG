import os
import torch
import argparse
import time
import statistics

import torch.nn as nn
import numpy as np
import torchvision.transforms as tfs
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from collections import defaultdict
from datasets.modelnet import FewShotSubModelNet, FewShotModelNet
from datasets.shapenet import FewShotShapeNet
from datasets.utils import EpisodicBatchSampler, SequentialBatchSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from metrics.evaluation_metrics import distChamferCUDA, emd_approx
from models.image_net import ImageEncoderWarpper

_modelnet_tfs = tfs.Compose([
    tfs.CenterCrop(550),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])

_shapenet_tfs = tfs.Compose([
    tfs.CenterCrop(256),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])

def to_cuda(in_dict):
    for eachKey in in_dict.keys():
        if eachKey in ['xs', 'xq', 'pcs', 'pcq']:
            in_dict[eachKey] = in_dict[eachKey].cuda()

# Training code for Few_shot Image2PC Protonet
def main(opt):
    # Load basic configuration parameters--------------------
    config_path, test_path, reference_path = opt.config_path, opt.test_path, opt.refer_path
    n_way, n_shot, n_episode = opt.n_way, opt.n_shot, opt.n_episode
    n_query = n_shot if opt.n_query == 0 else opt.n_query

    # Load Training & Testing Dataset----------------------------------
    # ds = FewShotModelNet(config_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_modelnet_tfs)
    # ds_test = FewShotModelNet(test_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_modelnet_tfs)
    ds_test = FewShotShapeNet(test_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_shapenet_tfs)

    # sampler = EpisodicBatchSampler(len(ds), n_way, n_episode)
    sampler_test = SequentialBatchSampler(len(ds_test))

    # dl = DataLoader(ds, batch_sampler=sampler, num_workers=0)  # num_workers=0 to avoid duplicate episode
    dl_test = DataLoader(ds_test, batch_sampler=sampler_test, num_workers=0)

    img_encoder = ImageEncoderWarpper(opt.img_encoder, finetune_layer=0)
    img_encoder = img_encoder.cuda()

    cd_list, emd_list = defaultdict(list), defaultdict(list)
    ttl_cd, ttl_emd = list(), list()
    for eachSample in tqdm(dl_test, desc='ONN Evluating'):
        
        to_cuda(eachSample)
        tmp_class_cd, tmp_class_emd = compute_onn(eachSample, nn_type='img', img_model=img_encoder)

        cd_list[eachSample['class'][0]].append(tmp_class_cd)
        ttl_cd.append(tmp_class_cd)
        emd_list[eachSample['class'][0]].append(tmp_class_emd)
        ttl_emd.append(tmp_class_emd)

    # Print results for every classes
    for eachKey in sorted(cd_list.keys()):
        tmp_logging = f'Class: {eachKey} -- Rec CD: {statistics.mean(cd_list[eachKey])} | Rec EMD {statistics.mean(emd_list[eachKey])} | Best CD: {min(cd_list[eachKey])} | Best EMD {min(emd_list[eachKey])} '
        print(tmp_logging)
        tmp_logging = f'Class: {eachKey} -- CD std: {statistics.stdev(cd_list[eachKey])} | EMD std {statistics.stdev(emd_list[eachKey])}'
        print(tmp_logging)


    print(f'Avg CD: {statistics.mean(ttl_cd)}({statistics.stdev(ttl_cd)}), Avg EMD: {statistics.mean(ttl_emd)}({statistics.stdev(ttl_emd)})')

def to_cuda(in_dict):
    for eachKey in in_dict.keys():
        if eachKey in ['xs', 'xq', 'pcs', 'pcq']:
            in_dict[eachKey] = in_dict[eachKey].cuda()

def compute_onn(sample, nn_type='pc', img_model=None):
    # print(sample.keys())
    xs, xq, pcs, pcq = Variable(sample['xs']), Variable(sample['xq']), Variable(sample['pcs']).squeeze(0), Variable(sample['pcq']).squeeze(0)
    n_class, n_support, n_query = 1, xs.size(1), xq.size(1)
    ttl_cd, ttl_emd = 0., 0.

    # print(pcq.shape)
    # print(pcs.shape)
    if nn_type == 'pc':
        for eachIdx in range(pcq.size(0)):
            ori_pc = pcq[eachIdx].unsqueeze(0).contiguous()
            # print(ori_pc.shape)

            tmp_pcs = pcq[eachIdx].unsqueeze(0).repeat(n_support, 1, 1)
            gen2gr, gr2gen = distChamferCUDA(ori_pc, pcq)
            dist = (gen2gr.mean(1) + gr2gen.mean(1))
            top1= dist.topk(1, largest=True)
            tmp_cd = top1.values
            top1_idx = top1.indices

            pred_pc = pcs[top1_idx].contiguous()
            tmp_emd = emd_approx(ori_pc, pred_pc)
            ttl_cd += tmp_cd.item()
            ttl_emd += tmp_emd.item()

    elif nn_type == 'img':
        img_corpus = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]), xq.view(n_class * n_query, *xq.size()[2:])], 0)
        img_z = img_model(img_corpus)
        img_z_dim = img_z.size(-1)
        img_zs, img_zq = img_z[:n_class * n_support], img_z[n_class * n_support:]

        for eachIdx in range(img_zq.size(0)):
            eachImgFeat = img_zq[eachIdx]
            tmp_feat = eachImgFeat.unsqueeze(0).repeat(n_support, 1)
            dist = torch.sum((tmp_feat - img_zs) ** 2, dim=1)

            top1_idx = dist.topk(1, largest=False).indices

            pred_pc, ori_pc =  pcs[top1_idx].contiguous(), pcq[eachIdx].unsqueeze(0).contiguous()
            gen2gr, gr2gen = distChamferCUDA(pred_pc, ori_pc)
            tmp_cd = (gen2gr.mean(1) + gr2gen.mean(1))
            tmp_emd = emd_approx(pred_pc, ori_pc)

            ttl_cd += tmp_cd.item()
            ttl_emd += tmp_emd.item()

    return ttl_cd / n_query, ttl_emd / n_query
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic parameters -- Data path, N-way-K-shot, etc.
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file: {DATASET}_{SPLIT}.txt')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test file: {DATASET}_{SPLIT}.txt')
    parser.add_argument('--refer_path', type=str, default='./extra_files/', help='Path to the reference folder')
    parser.add_argument('--n_way', type=int, default=1, help='Few shot setting: N-way')
    parser.add_argument('--n_shot', type=int, default=20, help='Few shot setting: K-shot')
    parser.add_argument('--n_query', type=int, default=0, help='Number of Query set, equals to --n_shot by default')

    # Network architecture parameters: Img_encoder, pc_encoder & pc_decoder
    parser.add_argument('--img_encoder', type=str, default='vgg_16', help='Image Encoder backbone')
    parser.add_argument('--pc_encoder', type=str, default='pointnet', help='Point Cloud Encoder backbone')
    parser.add_argument('--recon_factor', type=float, default=1.0, help='The weight of reconstruction loss')
    parser.add_argument('--intra_recon', action='store_true', help='Flag to enable intra-support set reconstruction')
    parser.add_argument('--epoch_start_recon', type=int, default=0, help='Epoch to start reconstruction task')
    parser.add_argument('--num_cluster', type=int, default=1, help='The number of MLP clusters of PC decoder')

    parser.add_argument('--num_slaves', type=int, default=16, help='PCDecoder parameter: number of MLP slaves (patches) per MLP cluster'),
    parser.add_argument('--device', type=str, default='cuda', help='PCDecoder parameter: cuda')
    parser.add_argument('--bottleneck_size', type=int, default=1536, help='PCDecoder parameter: Bottoleneck size Dim of img_feat + Dim of pc_feat')
    parser.add_argument('--num_layers', type=int, default=2, help='PCDecoder parameter: 2')
    parser.add_argument('--template_type', type=str, default="SQUARE", help='PCDecoder parameter: hidden sampling shape')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='PCDecoder parameter: number of hidden neurons')
    parser.add_argument('--activation', type=str, default='relu', help='PCDecoder parameter: activation function of PCDecoder')
    parser.add_argument('--dim_template', type=int, default=2, help='PCDecoder parameter: 2')
    parser.add_argument('--aggregate', type=str, default='mean', choices=['mean', 'full', 'mask_s', 'mask_m'])

    # Parameters for training:
    parser.add_argument('--n_episode', type=int, default=100, help='Number of episode per epoch')
    parser.add_argument('--epoch', type=int ,default=1000, help='Number of epochs to training (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=100, help='Decay learning rate every LR_DECAY epoches')
    parser.add_argument('--resume', type=int, default=-1, help='Flag to resume training')
    parser.add_argument('--pc_dist', type=str, default='cd', choices=['cd', 'emd'], help='The loss to train the network')
    parser.add_argument('--SGD', action='store_true', help='Flag to use SGD optimizer')

    # Experiment parameters: EXP_NAME, checkpoint path, etc.
    parser.add_argument('--name', type=str, default='0', help='Experiment Name')
    parser.add_argument('--dir_name', type=str, default='', help='Name of the log folder')
    parser.add_argument('--model_path', type=str, default='/home/yulin/Few_shot_point_cloud_reconstruction/checkpoint')
    parser.add_argument('--save_interval', type=int, default=50, help='Save Interval')
    parser.add_argument('--sample_interval', type=int, default=10, help='Sample Interval')
    parser.add_argument('--eval_interval', type=int, default=20, help='Evaluation Interval')

    conf = parser.parse_args()
    main(conf)


