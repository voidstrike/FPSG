import torch
import imageio
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .utils import euclidean_dist, build_pc_proto, emd_wrapper
# from metrics.evaluation_metrics import distChamferCUDA, distChamfer, emd_approx
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
    def __init__(self, img_encoder, pc_encoder, pc_decoder, mask_learner=None, recon_factor=1.0, metric='cd', intra_support=False, aggregate='single'):
        """
        Args:
            img_encoder (nn.Module): The image encoder
            pc_encoder (nn.Module): The point cloud encoder
            pc_decoder (nn.Module): The point cloud decoder
            mask_learner (nn.Module): An optional nn.Module that learns the prototype mask [default: None]
            recon_factor (float): The weight factor of loss between support & query set [default: 1]
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
        self.recon_factor = recon_factor 
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

        assert 1 == xs.size(0), 'Only support single-class for now'
        return self._loss_single_class(xs, xq, pcs, pcq)

    def _loss_single_class(self, img_s, img_q, pc_s, pc_q):
        # Single class reconstruct
        n_class, n_support, n_query = 1, img_s.size(1), img_q.size(1)

        ans = dict()

        # Combine support & query feature for faster inference
        img_corpus = torch.cat([img_s.view(n_class * n_support, *img_s.size()[2:]), img_q.view(n_class * n_query, *img_q.size()[2:])], 0)
        pc_corpus = pc_s.view(n_class * n_support, *pc_s.size()[2:]).transpose(2, 1)

        # Img feat
        img_z = self.img_encoder(img_corpus)
        img_z_dim = img_z.size(-1)
        img_zs, img_zq = img_z[:n_class * n_support], img_z[n_class * n_support:]

        # Compute class-specific point cloud prototype
        pc_z = self.pc_encoder(pc_corpus)
        pc_z_dim = pc_z.size(-1)
        pc_z_proto = pc_z.view(n_class, n_support, pc_z_dim)

        pc_z_proto = pc_z_proto.mean(1)  #  1 * 1 * 1024
        proto_mat_q = pc_z_proto.squeeze(0).repeat(n_query, 1)

        # ref_pc_q = pc_corpus.transpose(2, 1).contiguous()
        syn_pc = self.pc_decoder(torch.cat([img_zq, proto_mat_q], dim=1))

        ref_pc_q = pc_q.squeeze(0)
        loss_rec_q = self.pc_metric(syn_pc, ref_pc_q).sum()

        if self.intra_flag:
            proto_mat_s = pc_z_proto.squeeze(0).repeat(n_support, 1)
            syn_pc = self.pc_decoder(torch.cat([img_zs, proto_mat_s], dim=1))

            ref_pc_s = pc_s.squeeze(0)
            loss_rec_s = self.pc_metric(syn_pc, ref_pc_s).sum()
        else:
            loss_rec_s = _ZERO_HOLDER
            
        loss_recon = loss_rec_q + self.recon_factor * loss_rec_s
        ttl_loss = loss_recon

        ans['ttl_loss'] = ttl_loss
        ans['recon_loss'] = loss_recon
        ans['query_rec_loss'] = loss_rec_q
        ans['support_rec_loss'] = loss_rec_s

        return ans

    def _prototype_aggregate(self, proto_vec):
        if self.aggregate in ['single', 'mask_single']:
            proto_corpus = proto_vec.mean(1)  #  1 * 1 * 1024
        elif self.aggregate in ['multi', 'mask_multi']:
            proto_corpus = proto_vec          #  1 * NP * 1024 (NP=n_support)
        return proto_corpus

    def _build_final_input(self, img_z, pc_z):
        n_query = img_z.size(0)
        if self.aggregate == 'single':
            out_vec = pc_z.squeeze(0).repeat(n_query, 1).unsqueeze(1)
        elif self.aggregate == 'multi':
            out_vec = pc_z.repeat(n_query, 1, 1)
        elif self.aggregate == 'mask_single':
            tmp = pc_z.squeeze(0).repeat(n_query, 1)
            mask_vec = torch.cat([img_z, tmp], dim=1)
            mask_vec = self.mask_allocater(mask_vec)

            out_vec = (tmp * mask_vec).unsqueeze(1)
        else:
            n_np = pc_z.size(1)

            tmp_img = img_z.unsqueeze(1).repeat(1, n_np, 1).view(n_query * n_np, -1)
            tmp_pcz = pc_z.repeat(n_query, 1, 1).view(n_query * n_np, -1)

            mask_vec = torch.cat([tmp_img, tmp_pcz], dim=1)
            mask_vec = self.mask_allocater(mask_vec).view(n_query * n_np, -1)

            out_vec = (tmp_pcz * mask_vec).view(n_query, n_np, -1)

        return out_vec

    
    def draw_reconstruction(self, sample, img_path):
        xs, xq, pcs, pcq = Variable(sample['xs']), Variable(sample['xq']), Variable(sample['pcs']), Variable(sample['pcq'])
        n_class, n_support, n_query = xs.size(0), xs.size(1), xq.size(1)

        ans = dict()

        # Concatenate support set and query set for faster computation
        x = xq.view(n_class * n_query, *xq.size()[2:])

        pcs = pcs.view(n_class * n_support, *pcs.size()[2:]).transpose(2, 1)
        pcq = pcq.view(n_class * n_query, *pcq.size()[2:])

        img_z = self.img_encoder.forward(x)
        img_z_dim = img_z.size(-1)

        pc_z = self.pc_encoder(pcs)
        pc_z_dim = pc_z.size(-1)
        pc_z_proto = pc_z.view(n_class, n_support, pc_z_dim)

        pc_z_proto = self._prototype_aggregate(pc_z_proto)
        proto_mat_q = self._build_final_input(img_z, pc_z_proto)

        syn_pc = self.pc_decoder(img_z, proto_mat_q)

        image_list = list()
        tmp_idx = 0
        for each_gen_pc in syn_pc:
            image_list.append(visualize_point_clouds(each_gen_pc, pcq[tmp_idx], tmp_idx))
            tmp_idx += 1

        res = np.concatenate(image_list, axis=1)
        imageio.imwrite(img_path, res.transpose((1, 2, 0)))

    def get_pc_pairs(self, sample):
        xs, xq, pcs, pcq = Variable(sample['xs']), Variable(sample['xq']), Variable(sample['pcs']), Variable(sample['pcq'])
        n_class, n_support, n_query = xs.size(0), xs.size(1), xq.size(1)

        ans = dict()

        # Concatenate support set and query set for faster computation
        x = xq.view(n_class * n_query, *xq.size()[2:])

        pcs = pcs.view(n_class * n_support, *pcs.size()[2:]).transpose(2, 1)
        pcq = pcq.view(n_class * n_query, *pcq.size()[2:])

        img_z = self.img_encoder.forward(x)
        img_z_dim = img_z.size(-1)

        pc_z = self.pc_encoder(pcs)
        pc_z_dim = pc_z.size(-1)
        pc_z_proto = pc_z.view(n_class, n_support, pc_z_dim)

        pc_z_proto = self._prototype_aggregate(pc_z_proto)
        proto_mat_q = self._build_final_input(img_z, pc_z_proto)

        ref_pc_q = pcq.contiguous()
        syn_pc = self.pc_decoder(img_z, proto_mat_q)

        return syn_pc, ref_pc_q