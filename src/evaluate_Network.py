import os
import torch
import argparse
import time
import statistics

import torchvision.transforms as tfs

from tqdm import tqdm
from collections import defaultdict
from datasets.modelnet import FewShotModelNet
from datasets.shapenet import FewShotShapeNet
from datasets.utils import EpisodicBatchSampler, SequentialBatchSampler
from torch.utils.data import DataLoader

from models.few_shot import ImgPCProtoNet
from models.image_net import ImageEncoderWarpper
from models.point_cloud_net import PCEncoder, PCDecoder

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
    '''
    Auxiliary function that converts cpu tensors to gpu tensors
    '''
    for eachKey in in_dict.keys():
        if eachKey in ['xs', 'xq', 'xad', 'pcs', 'pcq', 'pcad']:
            in_dict[eachKey] = in_dict[eachKey].cuda()

def build_model(opt):
    '''
    Auxiliary function that creates model based on configuration
    '''
    img_encoder = ImageEncoderWarpper(opt.img_encoder, finetune_layer=3)  # Image branch
    pc_encoder = PCEncoder(opt.pc_encoder)  # Class-specific branch
    pc_decoder = PCDecoder(conf=opt)  # Literally, point cloud decoder

    # NOTE: legacy tets code, should be removed
    mask_allocater = None

    # Load pretrained pc_encoder, if any
    pc_encoder_dict_path = opt.pc_encoder_path
    if os.path.exists(pc_encoder_dict_path):
        print('Pretrained Model exist, loading')
        pc_encoder.load_state_dict(torch.load(pc_encoder_dict_path))

    # Class-agnostic branch is initialized inside the class definition
    model = ImgPCProtoNet(img_encoder, pc_encoder, pc_decoder, mask_learner=mask_allocater, query_factor=opt.query_factor, support_factor=opt.support_factor, intra_support=opt.intra_recon, aggregate=opt.aggregate)
    return model

# Evaluation code for Few_shot Image2PC Protonet
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

    # Load Testing Dataset----------------------------------
    if opt.dataset == 'modelnet':
        ds_test = FewShotModelNet(test_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_modelnet_tfs)
    elif opt.dataset == 'shapenet':
        ds_test = FewShotShapeNet(test_path, reference_path, n_classes=n_way, n_support=n_shot, n_query=n_query, transform=_shapenet_tfs)

    ran_sampler = EpisodicBatchSampler(len(ds_test), n_way, n_episode)
    seq_sampler = SequentialBatchSampler(len(ds_test))

    dl_test = DataLoader(ds_test, batch_sampler=seq_sampler if opt.sequential_eval else ran_sampler, num_workers=0)

    
    # Build Model -------------------------------------------
    model = build_model(opt)
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, opt.eval_model)))
    model = model.cuda()

    model.eval()
    tmp_idx = 0
    with torch.no_grad():
        ttl_rec_loss_q, ttl_rec_loss_s = 0., 0.
        acc_class_res_cd, acc_class_res_emd = defaultdict(list), defaultdict(list)
        acc_class_rec, acc_class_raw =  defaultdict(list), defaultdict(list)
        tmp_all_res = list()

        for sample in tqdm(dl_test, desc='Evaluating'):
            to_cuda(sample)
            # NOTE: following option are interchangeable, when using option 2, comment all lines till "pass"
            result_tuple = model._return_reconstruction(sample)  # OPTION 1
            # _ = model.draw_reconstruction(sample, [tmp_idx, opt.npy_folder])  # OPTION 2
            tmp_idx+=1

            tmp_class_cd = result_tuple['cd_loss'].item() / n_query
            tmp_class_emd = result_tuple['emd_loss'].item() / n_query

            acc_class_res_cd[sample['class'][0]].append(tmp_class_cd)
            acc_class_res_emd[sample['class'][0]].append(tmp_class_emd)

        # Print results for every classes
        for eachKey in sorted(acc_class_res_cd.keys()):
            tmp_logging = f'Class: {eachKey} -- Rec CD: {statistics.mean(acc_class_res_cd[eachKey])}; Rec EMD: {statistics.mean(acc_class_res_emd[eachKey])}'
            print(tmp_logging)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Basic parameters -- Data path, N-way-K-shot, etc.
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file: {DATASET}_{SPLIT}.txt;')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test file: {DATASET}_{SPLIT}.txt;')
    parser.add_argument('--refer_path', type=str, default='./modelnet_files/', help='Path to the reference folder [default: ./modelnet_files/];')
    parser.add_argument('--dataset', type=str, default='modelnet', choices=['modelnet', 'shapenet'], help='Type of training dataset [default: modelnet];')
    parser.add_argument('--pc_encoder_path', type=str, required=True, help='Path to the pre-trained pcencoder;')
    parser.add_argument('--n_way', type=int, default=1, help='Few shot setting: N-way [default: 1];')
    parser.add_argument('--n_shot', type=int, default=20, help='Few shot setting: K-shot [default: 20];')
    parser.add_argument('--n_query', type=int, default=0, help='Number of Query set [default: --n_shot];')

    # Network architecture parameters: Img_encoder, pc_encoder & pc_decoder
    parser.add_argument('--img_encoder', type=str, default='vgg_16', help='Image Encoder backbone')
    parser.add_argument('--pc_encoder', type=str, default='pointnet', help='Point Cloud Encoder backbone')
    parser.add_argument('--support_factor', type=float, default=1.0, help='The weight of support loss')
    parser.add_argument('--query_factor', type=float, default=1.0, help='The weight of query loss')
    parser.add_argument('--intra_recon', action='store_true', help='Flag to enable intra-support set reconstruction')
    parser.add_argument('--epoch_start_recon', type=int, default=0, help='Epoch to start reconstruction task')
    parser.add_argument('--num_clusters', type=int, default=4, help='The number of MLP clusters of PC decoder')
    parser.add_argument('--ori_dim', type=int, default=2, help='The dimension of the original surface [default: 2]')
    parser.add_argument('--raw_dim', type=int, default=3, help='The dimension of the deformed surface [default: 3]')

    parser.add_argument('--num_nodes', type=int, default=4, help='PCDecoder parameter: number of MLP slaves (patches) per MLP cluster'),
    parser.add_argument('--device', type=str, default='cuda', help='PCDecoder parameter: cuda')
    parser.add_argument('--bottleneck_size', type=int, default=1536, help='PCDecoder parameter: Bottoleneck size Dim of img_feat + Dim of pc_feat')
    parser.add_argument('--template_type', type=str, default="SQUARE", help='PCDecoder parameter: hidden sampling shape')
    parser.add_argument('--activation', type=str, default='relu', help='PCDecoder parameter: activation function of PCDecoder')
    parser.add_argument('--dim_template', type=int, default=2, help='PCDecoder parameter: 2')
    parser.add_argument('--aggregate', type=str, default='single', choices=['single', 'multi', 'mask_single', 'mask_multi'])

    # Parameters for training:
    parser.add_argument('--n_episode', type=int, default=100, help='Number of episode per epoch')
    parser.add_argument('--epoch', type=int ,default=500, help='Number of epochs to training (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=350, help='Decay learning rate every LR_DECAY epoches')
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
    parser.add_argument('--eval_model', type=str, required=True, help='Path to the pretrained Model')
    parser.add_argument('--sequential_eval', action='store_ture', help='Flag to evaluate model performance in sequential model')
    parser.add_argument('--npy_folder', type=str, default="./samples/", help='Path to store the generated point cloud & corresponding ground truth [default: ./samples/]')

    conf = parser.parse_args()
    main(conf)
