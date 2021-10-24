import torch
import imageio
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .utils import emd_wrapper
from .visualization import visualize_point_clouds

# Updated cd & emd implementation
from kaolin.metrics.pointcloud import chamfer_distance

_ZERO_HOLDER = torch.FloatTensor([0.]).cuda()
_AGGREGATOR = ['single', 'multi', 'mask_single', 'mask_multi']

# Main Network
class ImgPCProtoNet(nn.Module):
    """
    Core Universial Network
    """
    def __init__(self, img_encoder, pc_encoder, pc_decoder, mask_learner=None, query_factor=1.0, support_factor=1.0, metric='cd', intra_support=False, aggregate='single'):
        """
        Args:
            img_encoder (nn.Module): The image encoder
            pc_encoder (nn.Module): The point cloud encoder
            pc_decoder (nn.Module): The point cloud decoder
            mask_learner (nn.Module): An optional nn.Module that learns the prototype mask [default: None]
            query_factor (float): The weight factor of query set loss [default: 1]
            support_factor (float): The weight factor of AD support set loss [default: 1]
            metric (str): The training metric ('cd' or 'emd') [default: cd]
            intra_support (bool): The flag to use intra_support_training mode [default: False]
            aggregate (str): The aggregator of prototypes ('single', 'multi', 'mask_single', 'mask_multi') [default: 'single']
        """
        super(ImgPCProtoNet, self).__init__()

        # Network components
        self.img_encoder = img_encoder
        self.pc_encoder = pc_encoder
        self.pc_decoder = pc_decoder
        self.mask_allocater = mask_learner

        # Training parameters (factors & flags)
        self.query_factor = query_factor
        self.support_factor = support_factor
        self.intra_flag = intra_support

        if aggregate in _AGGREGATOR:
            self.aggregate = aggregate
        else:
            raise NotImplementedError(f'Found unsupported prototype aggragation: {aggregate}')

        # Loss functions
        if metric == 'cd':
            self.metric_module = None
            self.pc_metric = chamfer_distance
        elif metric == 'emd':
            self.pc_metric = self.emd_wrapper
        else:
            raise NotImplementedError(f'Found unsupported point cloud reconstruction metrics: {metric}')

    def loss(self, sample):
        # Gather input
        xs, xq, pcs, pcq = Variable(sample['xs']), Variable(sample['xq']), Variable(sample['pcs']), Variable(sample['pcq'])

        # NOTE: following 2 options are interchangeable

        # xad, pcad = xs, pcs  # Option 1
        xad, pcad = Variable(sample['xad']), Variable(sample['pcad'])  # Option 2
        

        return self._loss_single_class(xs, xq, xad, pcs, pcq, pcad)

    def _loss_single_class(self, img_s, img_q, img_ad, pc_s, pc_q, pc_ad):
        # 1-way-k-shot forward

        # Basic parameters
        n_support, n_query = img_s.size(1), img_q.size(1)
        ans = dict()

        # Combine support & query & aux feature for faster inference
        # Images -- ad set + query set
        img_corpus = torch.cat([
            img_ad.view(n_support, *img_ad.size()[2:]),
            img_q.view(n_query, *img_q.size()[2:]),
            ], dim=0)

        # Image features
        img_z = self.img_encoder(img_corpus)
        img_zad, img_zq = img_z[:n_support], img_z[n_support:]

        # Point clouds -- support set + ad set
        pc_corpus = torch.cat([
            pc_s.view(n_support, *pc_s.size()[2:]),
            pc_ad.view(n_support, *pc_ad.size()[2:]),
        ], dim=0).transpose(2, 1)
    
        # Point cloud features
        pc_z = self.pc_encoder(pc_corpus)
        pc_z_proto = pc_z[:n_support]
        pc_z_ad = pc_z[n_support:]

        pc_z_q = pc_z_proto.mean(0, keepdim=True)  #  class-specific shape prior feature
        proto_mat_q = pc_z_q.repeat(n_query, 1)

        syn_pc = self.pc_decoder(torch.cat([img_zq, proto_mat_q], dim=1))

        ref_pc_q = pc_q.squeeze(0)
        loss_rec_q = self.pc_metric(syn_pc, ref_pc_q).sum()

        if self.intra_flag:
            proto_mat_s = pc_z_ad
            syn_pc = self.pc_decoder(torch.cat([img_zad, proto_mat_s], dim=1))

            ref_pc_s = pc_ad.squeeze(0)
            loss_rec_s = self.pc_metric(syn_pc, ref_pc_s).sum()
        else:
            loss_rec_s = _ZERO_HOLDER
            
        loss_recon = self.query_factor * loss_rec_q + self.support_factor * loss_rec_s
        ttl_loss = loss_recon  # TODO: Legacy return issue

        ans['ttl_loss'] = ttl_loss
        ans['recon_loss'] = loss_recon
        ans['query_rec_loss'] = loss_rec_q
        ans['support_rec_loss'] = loss_rec_s

        return ans

    def _return_reconstruction(self, sample):
        img_s, img_q, pc_s, pc_q = Variable(sample['xs']), Variable(sample['xq']), Variable(sample['pcs']), Variable(sample['pcq'])
        img_ad, pc_ad = Variable(sample['xad']), Variable(sample['pcad'])
        n_class, n_support, n_query = 1, img_s.size(1), img_q.size(1)
        ans = dict()

        # Combine support & query & aux feature for faster inference
        # Imgs -- ad set + query set
        img_corpus = torch.cat([
            img_ad.view(n_support, *img_ad.size()[2:]),
            img_q.view(n_query, *img_q.size()[2:]),
            ], dim=0)

        # Img feat
        img_z = self.img_encoder(img_corpus)
        # img_z_dim = img_z.size(-1)
        img_zad, img_zq = img_z[:n_support], img_z[n_support:]

        # PCs -- support set + ad set
        pc_corpus = torch.cat([
            pc_s.view(n_support, *pc_s.size()[2:]),
            pc_ad.view(n_support, *pc_ad.size()[2:]),
        ], dim=0).transpose(2, 1)
    
        # PC feat
        pc_z = self.pc_encoder(pc_corpus)
        # pc_z_dim = pc_z.size(-1)
        pc_z_proto = pc_z[:n_support]
        pc_z_ad = pc_z[n_support:]

        pc_z_q = pc_z_proto.mean(0, keepdim=True)  #  1 * 1024
        proto_mat_q = pc_z_q.repeat(n_query, 1)

        syn_pc = self.pc_decoder(torch.cat([img_zq, proto_mat_q], dim=1))

        ref_pc_q = pc_q.squeeze(0)
        loss_rec_q = self.pc_metric(syn_pc, ref_pc_q).sum()
        emd_loss = emd_wrapper(syn_pc, ref_pc_q).sum()
        loss_recon = self.query_factor * loss_rec_q

        ans['cd_loss'] = loss_recon
        ans['emd_loss'] = emd_loss
        # ans['rec_pc'] = syn_pc
        # ans['raw_pc'] = pc_q

        return ans

    
    def draw_reconstruction(self, sample, img_path):
        xs, xq, pcs, pcq = Variable(sample['xs']), Variable(sample['xq']), Variable(sample['pcs']), Variable(sample['pcq'])
        n_class, n_support, n_query = xs.size(0), xs.size(1), xq.size(1)
        ori_tmp_code = sample['tmp'].item()

        # Concatenate support set and query set for faster computation
        x = xq.view(n_class * n_query, *xq.size()[2:])

        pcs = pcs.view(n_class * n_support, *pcs.size()[2:]).transpose(2, 1)
        pcq = pcq.squeeze(0)

        img_z = self.img_encoder.forward(x)
        img_z_dim = img_z.size(-1)

        pc_z = self.pc_encoder(pcs)
        pc_z_dim = pc_z.size(-1)
        pc_z_proto = pc_z.view(n_class, n_support, pc_z_dim)

        pc_z_proto = pc_z_proto.mean(1)  #  1 * 1 * 1024
        proto_mat_q = pc_z_proto.squeeze(0).repeat(n_query, 1)

        syn_pc = self.pc_decoder(torch.cat([img_z, proto_mat_q], dim=1))

        image_list = list()
        tmp_idx = 0
        for each_gen_pc in syn_pc:
            image_list.append(visualize_point_clouds(each_gen_pc, pcq[tmp_idx], tmp_idx))
            tmp_idx += 1

        res = np.concatenate(image_list, axis=1)
        imageio.imwrite(os.path.join(img_path[1], f'{img_path[0]}.png'), res.transpose((1, 2, 0)))
        npy_output = syn_pc.squeeze(0).cpu().detach().numpy()
        np.save(os.path.join(img_path[1], f'{img_path[0]}_{ori_tmp_code}.npy'), npy_output)
        npy_output = pcq.squeeze(0).cpu().detach().numpy()
        np.save(os.path.join(img_path[1], f'{img_path[0]}_{ori_tmp_code}_gt.npy'), npy_output)

