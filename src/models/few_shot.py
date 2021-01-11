import torch
import imageio
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from .utils import euclidean_dist, build_pc_proto
from metrics.evaluation_metrics import distChamferCUDA, distChamfer
from .visualization import visualize_point_clouds

_ZERO_HOLDER = torch.FloatTensor([0.]).cuda()

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ImgPCProtoNet(nn.Module):
    def __init__(self, img_encoder, pc_encoder, pc_decoder, mode='both', recon_factor=1.0, pc_metrics='cd', intra_support_training=False):
        super(ImgPCProtoNet, self).__init__()

        # Network component
        self.img_encoder = img_encoder
        self.pc_encoder = pc_encoder
        self.pc_decoder = pc_decoder

        # self.mode should be True if mode equals 'both', otherwise 'False'
        self.mode = True if mode == 'both' else False
        self.recon_factor = recon_factor if self.mode else 0.
        self.intra_flag = intra_support_training

        if pc_metrics == 'cd':
            self.pc_metric = distChamferCUDA
            # self.pc_metric = distChamfer
        else:
            raise NotImplementedError(f'Found unsupported point cloud reconstruction metrics: {pc_metrics}')

    def loss(self, sample):
        # Sample is a tuple
        xs = Variable(sample['xs']) # Support set -- Imgs WAY * SHOT * C * H * W
        xq = Variable(sample['xq']) # Query set -- Imgs
        pcs = Variable(sample['pcs']) # Support set -- Pcs WAY * SHOT * 2048 * 3 -> Should convert to 3 * 2048
        pcq = Variable(sample['pcq']) # Query set -- Pcs

        n_class = xs.size(0)
        assert xq.size(0) == n_class, f'Inconsistent number of classes between support set ({n_class}) and query set ({xq.size(0)})'
        n_support, n_query = xs.size(1), xq.size(1)

        ans = dict()

        tgt_idx_q = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        tgt_idx_q = Variable(tgt_idx_q, requires_grad=False)

        # Pending usage for intra-support set training
        tgt_idx_s = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_support, 1).long()
        tgt_idx_s = Variable(tgt_idx_s.reshape(n_class * n_support, 1), requires_grad=False)

        if xq.is_cuda:
            tgt_idx_q = tgt_idx_q.cuda()
            tgt_idx_s = tgt_idx_s.cuda()

        # Concatenate support set and query set for faster computation
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]), xq.view(n_class * n_query, *xq.size()[2:])], 0)
        pc = torch.cat([pcs.view(n_class * n_support, *pcs.size()[2:]), pcq.view(n_class * n_query, *pcq.size()[2:])], 0)
        pc = pc.transpose(2, 1)

        img_z = self.img_encoder.forward(x)
        img_z_dim = img_z.size(-1)

        img_z_proto = img_z[:n_class * n_support].view(n_class, n_support, img_z_dim).mean(1)
        img_zs = img_z[:n_class * n_support]
        img_zq = img_z[n_class * n_support:]

        dists = euclidean_dist(img_zq, img_z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        # Compute Loss for 2D information, L_{CE}
        loss_clf = -log_p_y.gather(2, tgt_idx_q).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, tgt_idx_q.squeeze()).float().mean()

        if self.mode:
            # Enable 3D Reconstruction sub task
            pc_z = self.pc_encoder(pc)
            pc_z_dim = pc_z.size(-1)

            pc_z_proto = pc_z[:n_class * n_support].view(n_class, n_support, pc_z_dim).mean(1) # Get few-shot 3D prototype
            pcq = pc[n_class * n_support:].contiguous()

            y_hat = y_hat.view(n_class * n_query, 1)

            proto_mat_q = build_pc_proto(n_class, y_hat, pc_z_proto).unsqueeze(1)
            syn_pc = self.pc_decoder(img_zq, proto_mat_q)

            gen2gr, gr2gen = self.pc_metric(syn_pc, pcq)
            loss_rec_q = (gen2gr.mean(1) + gr2gen.mean(1)).sum()

            if self.intra_flag:
                pcs = pc[:n_class * n_support].contiguous()

                proto_mat_s = build_pc_proto(n_class, tgt_idx_s, pc_z_proto).unsqueeze(1)
                syn_pc = self.pc_decoder(img_zs, proto_mat_s)

                gen2gr, gr2gen = self.pc_metric(syn_pc, pcs)
                loss_rec_s = (gen2gr.mean(1) + gr2gen.mean(1)).sum()
            else:
                loss_rec_s = 0.
            
            loss_recon = loss_rec_q + loss_rec_s
        else:
            loss_recon = _ZERO_HOLDER

        ttl_loss = loss_clf + self.recon_factor * loss_recon

        ans['ttl_loss'] = ttl_loss
        ans['cls_acc'] = acc_val
        ans['cls_loss'] = loss_clf
        ans['recon_loss'] = loss_recon

        return ans

    
    def draw_reconstruction(self, sample, img_path):
        # Sample is a tuple
        xs = Variable(sample['xs']) # Support set -- Imgs WAY * SHOT * C * H * W
        xq = Variable(sample['xq']) # Query set -- Imgs
        pcs = Variable(sample['pcs']) # Support set -- Pcs WAY * SHOT * 2048 * 3 -> Should convert to 3 * 2048
        pcq = Variable(sample['pcq']) # Query set -- Pcs

        n_class = xs.size(0)
        assert xq.size(0) == n_class, f'Inconsistent number of classes between support set ({n_class}) and query set ({xq.size(0)})'
        n_support, n_query = xs.size(1), xq.size(1)

        ans = dict()

        tgt_idx_q = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        tgt_idx_q = Variable(tgt_idx_q, requires_grad=False)

        # Pending usage for intra-support set training
        tgt_idx_s = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_support, 1).long()
        tgt_idx_s = Variable(tgt_idx_s, requires_grad=False)

        if xq.is_cuda:
            tgt_idx_q = tgt_idx_q.cuda()
            tgt_idx_s = tgt_idx_s.cuda()

        # Concatenate support set and query set for faster computation
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]), xq.view(n_class * n_query, *xq.size()[2:])], 0)

        pcs = pcs.view(n_class * n_support, *pcs.size()[2:]).transpose(2, 1)
        pcq = pcq.view(n_class * n_query, *pcq.size()[2:])

        img_z = self.img_encoder.forward(x)
        img_z_dim = img_z.size(-1)

        img_z_proto = img_z[:n_class * n_support].view(n_class, n_support, img_z_dim).mean(1)
        img_zs = img_z[:n_class * n_support]
        img_zq = img_z[n_class * n_support:]

        # Get predicted label
        dists = euclidean_dist(img_zq, img_z_proto)
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
        _, y_hat = log_p_y.max(2)

        # Enable 3D Reconstruction sub task
        pc_z = self.pc_encoder(pcs)
        pc_z_dim = pc_z.size(-1)

        pc_z_proto = pc_z.view(n_class, n_support, pc_z_dim).mean(1) # Get few-shot 3D prototype

        y_hat = y_hat.view(n_class * n_query, 1)
        proto_mat_q = build_pc_proto(n_class, y_hat, pc_z_proto).unsqueeze(1)

        syn_pc = self.pc_decoder(img_zq, proto_mat_q)

        image_list = list()
        tmp_idx = 0
        for each_gen_pc in syn_pc:
            image_list.append(visualize_point_clouds(each_gen_pc, pcq[tmp_idx], tmp_idx))
            tmp_idx += 1

        res = np.concatenate(image_list, axis=1)
        imageio.imwrite(img_path, res.transpose((1, 2, 0)))
        pass