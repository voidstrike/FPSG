import os
import torch
import argparse

import torch.nn as nn
import numpy as np
import torchvision.transforms as tfs
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
from datasets.modelnet import FewShotSubModelNet, FewShotModelNet
from datasets.utils import EpisodicBatchSampler
from torch.utils.data import DataLoader

from pointnet.model import PointNetfeat
from models.few_shot import ImgPCProtoNet
from models.image_net import ImageEncoderWarpper
from models.point_cloud_net import PCEncoder, PCDecoder

_modelnet_tfs = tfs.Compose([
    tfs.CenterCrop(550),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])

def to_cuda(in_dict):
    for eachKey in in_dict.keys():
        if eachKey in ['xs', 'xq', 'pcs', 'pcq']:
            in_dict[eachKey] = in_dict[eachKey].cuda()
            # print(f'{eachKey}: {in_dict[eachKey].shape}')

# Training code for Few_shot Image2PC Protonet
def main(opt):
    # Load basic configuration parameters--------------------
    config_path, test_path, reference_path = opt.config_path, opt.test_path, opt.refer_path
    n_way, n_shot, n_episode = opt.n_way, opt.n_shot, opt.n_episode
    n_query = n_shot if opt.n_query == 0 else opt.n_query

    # Make checkpoint folder
    checkpoint_path = os.path.join(opt.model_path, opt.name)
    checkpoint_imgs = os.path.join(checkpoint_path, 'images')

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    if not os.path.exists(checkpoint_imgs):
        os.mkdir(checkpoint_imgs)

    # Load Training & Testing Dataset----------------------------------
    ds = FewShotModelNet(config_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_modelnet_tfs)
    ds_test = FewShotModelNet(test_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_modelnet_tfs)
    sampler = EpisodicBatchSampler(len(ds), n_way, n_episode)
    sampler_test = EpisodicBatchSampler(len(ds_test), n_way, n_episode)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=0)  # num_workers=0 to avoid duplicate episode
    dl_test = DataLoader(ds_test, batch_sampler=sampler_test, num_workers=0)

    # Build Model -------------------------------------------
    img_encoder = ImageEncoderWarpper(opt.img_encoder, finetune_layer=3)
    pc_encoder = PCEncoder(opt.pc_encoder)
    pc_decoder = PCDecoder(configuration=opt)

    # Load PCEncoder------------------------------------------
    pc_encoder_dict_path = opt.pc_encoder_path
    pc_encoder.load_state_dict(torch.load(pc_encoder_dict_path))

    model = ImgPCProtoNet(img_encoder, pc_encoder, pc_decoder, recon_factor=opt.recon_factor, intra_support_training=opt.intra_recon)

    # Load previous model if applicable
    start_epoch = 1
    if opt.resume > 0:
        start_epoch = opt.resume
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, f'model_epoch_{start_epoch}.pt')))

    model = model.cuda()

    # Pre-training configuration------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.lr,
        betas=(.9, .999)
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay, gamma=.5)

    # Start Training process----------------------------------
    model.train()
    model.mode = False
    for epoch in range(start_epoch, opt.epoch+1):
        ttl_clf_loss = 0.0
        ttl_rec_loss = 0.0
        ttl_clf_acc = 0.0

        if epoch >= opt.epoch_start_recon:
            model.mode = True

        # MAIN TRAINING CODE
        for sample in tqdm(dl, desc='Epoch {:d} training'.format(epoch)):
            to_cuda(sample)
            optimizer.zero_grad()
            result_tuple = model.loss(sample)
            result_tuple['ttl_loss'].backward()
            optimizer.step()

            ttl_clf_loss += result_tuple['cls_loss'].item()
            ttl_clf_acc += result_tuple['cls_acc'].item()
            ttl_rec_loss += result_tuple['recon_loss'].item()

        print(f'Training Results for this epoch are: L_clf: {ttl_clf_loss / n_episode}, L_rec: {ttl_rec_loss / n_episode}, CLF_ACC: {ttl_clf_acc / n_episode}')

        if epoch >= opt.epoch_start_recon:
            scheduler.step()
        
        # SAVE 
        if epoch % opt.save_interval == 0 or epoch == opt.epoch:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'model_epoch_{epoch}.pt'))

        # VISUALIZE
        if epoch % opt.sample_interval == 0 and epoch >= opt.epoch_start_recon:
            with torch.no_grad():
                for sample in dl:
                    to_cuda(sample)
                    model.draw_reconstruction(sample, os.path.join(checkpoint_imgs, f'sample_img_{epoch}.png'))
                    break

        # EVALUATION
        if epoch % opt.eval_interval == 0 or epoch == opt.epoch:
            tmp_clf_loss = 0.0
            tmp_rec_loss = 0.0
            tmp_clf_acc = 0.0

            with torch.no_grad():
                for sample in tqdm(dl_test, desc='Epoch {:d} training'.format(epoch)):
                    to_cuda(sample)
                    result_tuple = model.loss(sample)

                    tmp_clf_loss += result_tuple['cls_loss'].item()
                    tmp_clf_acc += result_tuple['cls_acc'].item()
                    tmp_rec_loss += result_tuple['recon_loss'].item()

                for sample in dl_test:
                    to_cuda(sample)
                    model.draw_reconstruction(sample, os.path.join(checkpoint_imgs, f'sample_img_{epoch}_test.png'))
                    break

            print(f'Testing Results for this epoch are: L_clf: {tmp_clf_loss / n_episode}, L_rec: {tmp_rec_loss / n_episode}, CLF_ACC: {tmp_clf_acc / n_episode}')
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic parameters -- Data path, N-way-K-shot, etc.
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file: {DATASET}_{SPLIT}.txt')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test file: {DATASET}_{SPLIT}.txt')
    parser.add_argument('--refer_path', type=str, default='./extra_files/', help='Path to the reference folder')
    parser.add_argument('--pc_encoder_path', type=str, required=True, help='Path to the pre-trained pcencoder')
    parser.add_argument('--n_way', type=int, default=10, help='Few shot setting: N-way')
    parser.add_argument('--n_shot', type=int, default=5, help='Few shot setting: K-shot')
    parser.add_argument('--n_query', type=int, default=0, help='Number of Query set, equals to --n_shot by default')

    # Network architecture parameters: Img_encoder, pc_encoder & pc_decoder
    parser.add_argument('--img_encoder', type=str, default='vgg_16', help='Image Encoder backbone')
    parser.add_argument('--pc_encoder', type=str, default='pointnet', help='Point Cloud Encoder backbone')
    parser.add_argument('--recon_factor', type=float, default=1.0, help='The weight of reconstruction loss')
    parser.add_argument('--intra_recon', action='store_true', help='Flag to enable intra-support set reconstruction')
    parser.add_argument('--epoch_start_recon', type=int, default=50, help='Epoch to start reconstruction task')

    parser.add_argument('--num_slaves', type=int, default=8, help='PCDecoder parameter: number of MLP slaves (patches) per MLP cluster'),
    parser.add_argument('--device', type=str, default='cuda', help='PCDecoder parameter: cuda')
    parser.add_argument('--bottleneck_size', type=int, default=1536, help='PCDecoder parameter: Bottoleneck size Dim of img_feat + Dim of pc_feat')
    parser.add_argument('--num_layers', type=int, default=2, help='PCDecoder parameter: 2')
    parser.add_argument('--template_type', type=str, default="SQUARE", help='PCDecoder parameter: hidden sampling shape')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='PCDecoder parameter: number of hidden neurons')
    parser.add_argument('--activation', type=str, default='relu', help='PCDecoder parameter: activation function of PCDecoder')
    parser.add_argument('--dim_template', type=int, default=2, help='PCDecoder parameter: 2')

    # Parameters for training:
    parser.add_argument('--n_episode', type=int, default=100, help='Number of episode per epoch')
    parser.add_argument('--epoch', type=int ,default=1000, help='Number of epochs to training (default: 10000)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=50, help='Decay learning rate every LR_DECAY epoches')
    parser.add_argument('--resume', type=int, default=-1, help='Flag to resume training')

    # Experiment parameters: EXP_NAME, checkpoint path, etc.
    parser.add_argument('--name', type=str, default='0', help='Experiment Name')
    parser.add_argument('--dir_name', type=str, default='', help='Name of the log folder')
    parser.add_argument('--model_path', type=str, default='/home/yulin/Few_shot_point_cloud_reconstruction/checkpoint')
    parser.add_argument('--save_interval', type=int, default=50, help='Save Interval')
    parser.add_argument('--sample_interval', type=int, default=10, help='Sample Interval')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation Interval')

    conf = parser.parse_args()
    main(conf)
