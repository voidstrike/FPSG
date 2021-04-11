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

from pointnet.model import PointNetfeat
from models.few_shot import ImgPCProtoNet
from models.image_net import ImageEncoderWarpper
from models.point_cloud_net import PCEncoder, PCDecoder
from models.support_models import FCMaskAlloacter, TransMaskAllocater

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

    # Make checkpoint folder
    timestamp = time.strftime('%m_%d_%H_%M')
    checkpoint_path = os.path.join(opt.model_path, opt.name)
    checkpoint_imgs = os.path.join(checkpoint_path, 'images')
    checkpoint_logs = os.path.join(checkpoint_path, f'log_{timestamp}.txt')

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.exists(checkpoint_imgs):
        os.mkdir(checkpoint_imgs)

    # Load Training & Testing Dataset----------------------------------
    if opt.dataset == 'modelnet':
        ds = FewShotModelNet(config_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_modelnet_tfs)
        ds_test = FewShotModelNet(test_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_modelnet_tfs)
    elif opt.dataset == 'shapenet':
        ds = FewShotShapeNet(config_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_shapenet_tfs)
        ds_test = FewShotShapeNet(test_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_shapenet_tfs)

    sampler = EpisodicBatchSampler(len(ds), n_way, n_episode)
    tmp_sampler = EpisodicBatchSampler(len(ds_test), n_way, n_episode)
    sampler_test = SequentialBatchSampler(len(ds_test))

    dl = DataLoader(ds, batch_sampler=sampler, num_workers=0)  # num_workers=0 to avoid duplicate episode
    dl_test = DataLoader(ds_test, batch_sampler=sampler_test, num_workers=0)
    # dl_test = DataLoader(ds_test, batch_sampler=tmp_sampler, num_workers=0)

    # Build Model -------------------------------------------

    model = build_model(opt)

    # Load previous model if applicable
    start_epoch = 1
    if opt.resume > 0:
        print(f'Resume previous training, start from epoch {opt.resume}, loading previous model')
        start_epoch = opt.resume
        resume_model_path = os.path.join(checkpoint_path, f'model_epoch_{start_epoch}.pt')

        if os.path.exists(resume_model_path):
            model.load_state_dict(torch.load(resume_model_path))
        else:
            raise RuntimeError(f'{resume_model_path} does not exist, loading failed')

    model = model.cuda()

    # Optimizer & lr_scheduler configuration------------------------------
    if not opt.SGD:
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt.lr,
            betas=(.9, .999)
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt.lr,
            weight_decay=1e-2
        )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay, gamma=.5)

    # Start Training process----------------------------------
    model.train()
    log_content = list()

    for epoch in range(start_epoch, opt.epoch + 1):
        ttl_rec_loss_q, ttl_rec_loss_s = 0., 0.

        # MAIN TRAINING CODE
        for sample in tqdm(dl, desc='Epoch {:d} training'.format(epoch)):
            to_cuda(sample)
            optimizer.zero_grad()
            result_tuple = model.loss(sample)
            result_tuple['ttl_loss'].backward()
            optimizer.step()

            ttl_rec_loss_q += result_tuple['query_rec_loss'].item() / n_query
            ttl_rec_loss_s += result_tuple['support_rec_loss'].item() / n_shot

        tmp_logging = f'Training Results for Epoch -- {epoch} are: Query_rec: {ttl_rec_loss_q / n_episode}, Support_rec: {ttl_rec_loss_s / n_episode}'
        log_content.append(tmp_logging)
        print(tmp_logging)

        scheduler.step()

        # EVALUATION
        if epoch % opt.eval_interval == 0 or epoch == opt.epoch:
        # if True:
            test_rec_q, test_rec_s = 0., 0.
            test_emd_q = 0.
            acc_class_res_cd = defaultdict(list)
            acc_class_res_emd = defaultdict(list)
            tmp_all_res = list()

            with torch.no_grad():
                for sample in tqdm(dl_test, desc='Epoch {:d} evaluating'.format(epoch)):
                    to_cuda(sample)
                    result_tuple = model.loss(sample, emd_flag=True)

                    tmp_class_cd = result_tuple['query_rec_loss'].item() / n_query
                    tmp_class_emd = result_tuple['query_emd'].item() / n_query
                    test_rec_q += tmp_class_cd
                    test_emd_q += tmp_class_emd
                    test_rec_s += result_tuple['support_rec_loss'].item() / n_shot

                    acc_class_res_cd[sample['class'][0]].append(tmp_class_cd)
                    acc_class_res_emd[sample['class'][0]].append(tmp_class_emd)
                    tmp_all_res.append(tmp_class_cd)

                # Print results for every classes
                for eachKey in sorted(acc_class_res_cd.keys()):
                    tmp_logging = f'Class: {eachKey} -- Rec CD: {statistics.mean(acc_class_res_cd[eachKey])} ({statistics.stdev(acc_class_res_cd[eachKey])})'
                    print(tmp_logging)
                    log_content.append(tmp_logging)

                for sample in dl_test:
                    to_cuda(sample)
                    model.draw_reconstruction(sample, os.path.join(checkpoint_imgs, f'sample_img_{epoch}_test.png'))
                    break

            tmp_logging = f'Avg testing results across all classes Epoch -- {epoch} are: Query_rec: {test_rec_q / len(ds_test)} ({statistics.stdev(tmp_all_res)})'
            log_content.append(tmp_logging)
            print(tmp_logging)
        
        # SAVE 
        if epoch % opt.save_interval == 0 or epoch == opt.epoch:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'model_epoch_{epoch}.pt'))
            # Writing log file
            with open(checkpoint_logs, 'a') as f:
                f.writelines(f'{line}\n' for line in log_content)
            log_content = list()

        # VISUALIZE -- first epoch only
        if epoch % opt.sample_interval == 0:
            with torch.no_grad():
                for sample in dl:
                    to_cuda(sample)
                    model.draw_reconstruction(sample, os.path.join(checkpoint_imgs, f'sample_img_{epoch}.png'))
                    break
    pass

def build_model(opt):
    # Build Model -------------------------------------------
    img_encoder = ImageEncoderWarpper(opt.img_encoder, finetune_layer=3)
    pc_encoder = PCEncoder(opt.pc_encoder)
    pc_decoder = PCDecoder(conf=opt)

    mask_allocater = None
    if opt.aggregate in ['mask_s', 'mask_m']:
        mask_allocater = FCMaskAlloacter(512, 1024)

    # Load PCEncoder------------------------------------------
    pc_encoder_dict_path = opt.pc_encoder_path
    if os.path.exists(pc_encoder_dict_path):
        print('Pretrained Model exist, loading')
        pc_encoder.load_state_dict(torch.load(pc_encoder_dict_path))

    model = ImgPCProtoNet(img_encoder, pc_encoder, pc_decoder, mask_learner=mask_allocater,recon_factor=opt.recon_factor, intra_support=opt.intra_recon, aggregate=opt.aggregate)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic parameters -- Data path, N-way-K-shot, etc.
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file: {DATASET}_{SPLIT}.txt')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test file: {DATASET}_{SPLIT}.txt')
    parser.add_argument('--refer_path', type=str, default='./extra_files/', help='Path to the reference folder')
    parser.add_argument('--dataset', type=str, default='modelnet', choices=['modelnet', 'shapenet'], help='Type of training dataset')
    parser.add_argument('--pc_encoder_path', type=str, required=True, help='Path to the pre-trained pcencoder')
    parser.add_argument('--n_way', type=int, default=1, help='Few shot setting: N-way')
    parser.add_argument('--n_shot', type=int, default=20, help='Few shot setting: K-shot')
    parser.add_argument('--n_query', type=int, default=0, help='Number of Query set, equals to --n_shot by default')

    # Network architecture parameters: Img_encoder, pc_encoder & pc_decoder
    parser.add_argument('--img_encoder', type=str, default='vgg_16', help='Image Encoder backbone')
    parser.add_argument('--pc_encoder', type=str, default='pointnet', help='Point Cloud Encoder backbone')
    parser.add_argument('--recon_factor', type=float, default=1.0, help='The weight of reconstruction loss')
    parser.add_argument('--intra_recon', action='store_true', help='Flag to enable intra-support set reconstruction')
    parser.add_argument('--epoch_start_recon', type=int, default=0, help='Epoch to start reconstruction task')
    parser.add_argument('--num_clusters', type=int, default=4, help='The number of MLP clusters of PC decoder')
    parser.add_argument('--ori_dim', type=int, default=2, help='The dimension of the original surface [default: 2]')
    parser.add_argument('--raw_dim', type=int, default=3, help='The dimension of the deformed surface [default: 3]')

    parser.add_argument('--num_nodes', type=int, default=4, help='PCDecoder parameter: number of MLP slaves (patches) per MLP cluster'),
    parser.add_argument('--device', type=str, default='cuda', help='PCDecoder parameter: cuda')
    parser.add_argument('--bottleneck_size', type=int, default=1536, help='PCDecoder parameter: Bottoleneck size Dim of img_feat + Dim of pc_feat')
    # parser.add_argument('--num_layers', type=int, default=2, help='PCDecoder parameter: 2')
    parser.add_argument('--template_type', type=str, default="SQUARE", help='PCDecoder parameter: hidden sampling shape')
    # parser.add_argument('--hidden_neurons', type=int, default=512, help='PCDecoder parameter: number of hidden neurons')
    parser.add_argument('--activation', type=str, default='relu', help='PCDecoder parameter: activation function of PCDecoder')
    parser.add_argument('--dim_template', type=int, default=2, help='PCDecoder parameter: 2')
    parser.add_argument('--aggregate', type=str, default='single', choices=['single', 'multi', 'mask_single', 'mask_multi'])

    # Parameters for training:
    parser.add_argument('--n_episode', type=int, default=100, help='Number of episode per epoch')
    parser.add_argument('--epoch', type=int ,default=10000, help='Number of epochs to training (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=200, help='Decay learning rate every LR_DECAY epoches')
    parser.add_argument('--resume', type=int, default=-1, help='Flag to resume training')
    parser.add_argument('--pc_dist', type=str, default='cd', choices=['cd', 'emd'], help='The loss to train the network')
    parser.add_argument('--SGD', action='store_true', help='Flag to use SGD optimizer')

    # Experiment parameters: EXP_NAME, checkpoint path, etc.
    parser.add_argument('--name', type=str, default='0', help='Experiment Name')
    parser.add_argument('--dir_name', type=str, default='', help='Name of the log folder')
    parser.add_argument('--model_path', type=str, default='../checkpoint')
    parser.add_argument('--save_interval', type=int, default=50, help='Save Interval')
    parser.add_argument('--sample_interval', type=int, default=10, help='Sample Interval')
    parser.add_argument('--eval_interval', type=int, default=20, help='Evaluation Interval')
    parser.add_argument('--eval_model', type=str, default='NONE', help='Path to the pretrained Model')

    conf = parser.parse_args()
    main(conf)
