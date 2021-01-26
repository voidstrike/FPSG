import torch
import imageio
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .utils import euclidean_dist, build_pc_proto
from metrics.evaluation_metrics import distChamferCUDA, distChamfer, emd_approx
from .visualization import visualize_point_clouds

_ZERO_HOLDER = torch.FloatTensor([0.]).cuda()
_PRESET_AGGREGATE = ['mean', 'full', 'mask_s', 'mask_m']

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# Main Network
class ImgPCProtoNet(nn.Module):
    def __init__(self, img_encoder, pc_encoder, pc_decoder, mask_allocater=None, recon_factor=1.0, pc_metrics='cd', intra_support_training=False, aggregate='mean'):
        super(ImgPCProtoNet, self).__init__()

        # Network components
        self.img_encoder = img_encoder
        self.pc_encoder = pc_encoder
        self.pc_decoder = pc_decoder
        self.mask_allocater = mask_allocater

        # Training parameters (factors & flags)
        self.recon_factor = recon_factor 
        self.intra_flag = intra_support_training

        if aggregate in _PRESET_AGGREGATE:
            self.aggregate = aggregate
        else:
            raise NotImplementedError(f'Found unsupported prototype aggragation: {aggregate}')

        # Loss functions
        if pc_metrics == 'cd':
            self.pc_metric = distChamferCUDA
        elif pc_metrics == 'emd':
            self.pc_metric = emd_approx
        else:
            raise NotImplementedError(f'Found unsupported point cloud reconstruction metrics: {pc_metrics}')

    def loss(self, sample, emd_flag=False):
        # Gather input
        xs, xq, pcs, pcq = Variable(sample['xs']), Variable(sample['xq']), Variable(sample['pcs']), Variable(sample['pcq'])

        if 1 == xs.size(0):
            return self._loss_single_class(xs, xq, pcs, pcq, emd_flag=emd_flag)
        else:
            return self._loss_multiple_class(xs, xq, pcs, pcq)

    def _loss_single_class(self, img_s, img_q, pc_s, pc_q, emd_flag=False):
        # Single class reconstruct
        n_class, n_support, n_query = 1, img_s.size(1), img_q.size(1)

        ans = dict()

        # Prepare features 
        img_corpus = torch.cat([img_s.view(n_class * n_support, *img_s.size()[2:]), img_q.view(n_class * n_query, *img_q.size()[2:])], 0)
        pc_corpus = pc_s.view(n_class * n_support, *pc_s.size()[2:]).transpose(2, 1)

        # Img feat
        img_z = self.img_encoder(img_corpus)
        img_z_dim = img_z.size(-1)
        img_zs, img_zq = img_z[:n_class * n_support], img_z[n_class * n_support:]

        # PC feat
        pc_z = self.pc_encoder(pc_corpus)
        pc_z_dim = pc_z.size(-1)
        pc_z_proto = pc_z.view(n_class, n_support, pc_z_dim)

        pc_z_proto = self._prototype_aggregate(pc_z_proto)
        proto_mat_q = self._build_final_input(img_zq, pc_z_proto)

        ref_pc_q = pc_corpus.transpose(2, 1).contiguous()
        syn_pc = self.pc_decoder(img_zq, proto_mat_q)

        gen2gr, gr2gen = self.pc_metric(syn_pc, ref_pc_q)
        loss_rec_q = (gen2gr.mean(1) + gr2gen.mean(1)).sum()

        if emd_flag:
            emd_rec_q = emd_approx(syn_pc, ref_pc_q).sum()
            ans['query_emd'] = emd_rec_q

        if self.intra_flag:
            proto_mat_s = self._build_final_input(img_zs, pc_z_proto)

            ref_pc_s = pc_s
            syn_pc = self.pc_decoder(img_zs, proto_mat_s)

            gen2gr, gr2gen = self.pc_metric(syn_pc, ref_pc_s)
            loss_rec_s = (gen2gr.mean(1) + gr2gen.mean(1)).sum()
        else:
            loss_rec_s = _ZERO_HOLDER
            
        loss_recon = loss_rec_q + self.recon_factor * loss_rec_s
        ttl_loss = loss_recon

        ans['ttl_loss'] = ttl_loss
        ans['recon_loss'] = loss_recon
        ans['query_rec_loss'] = loss_rec_q
        ans['support_rec_loss'] = loss_rec_s

        return ans

    def _loss_multiple_class(self, img_s, img_q, pc_s, pc_q):
        pass

    def _prototype_aggregate(self, proto_vec):
        if self.aggregate in ['mean', 'mask_s']:
            proto_corpus = proto_vec.mean(1)  #  1 * 1 * 1024
        elif self.aggregate in ['full', 'mask_m']:
            proto_corpus = proto_vec          #  1 * NP * 1024 (NP=n_support)
        return proto_corpus

    def _build_final_input(self, img_z, pc_z):
        n_query = img_z.size(0)
        if self.aggregate == 'mean':
            out_vec = pc_z.squeeze(0).repeat(n_query, 1).unsqueeze(1)
        elif self.aggregate == 'full':
            out_vec = pc_z.repeat(n_query, 1, 1)
        elif self.aggregate == 'mask_s':
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