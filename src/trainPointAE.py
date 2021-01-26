import argparse
import torch
import os

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torchvision.transforms as tfs

from datasets.mv_dataset import MultiViewDataSet
from metrics.evaluation_metrics import distChamferCUDA
from pointnet.model import PointNetfeat
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.point_cloud_net import PCEncoder
from models.support_models import AuxClassifier
from tqdm import tqdm

# Dummy transformation for MV_DATASET
_modelnet_tfs = tfs.Compose([
    tfs.CenterCrop(550),
    tfs.Resize(224),
    tfs.ToTensor(),
    tfs.Normalize((.5, .5, .5), (.5, .5, .5))
])

def main(opt):
    # Load Basic configuration
    checkpoint_path = os.path.join(opt.model_path, opt.name)
    root, ply_root = opt.root, opt.proot

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Load Datasets
    mv_ds = MultiViewDataSet(root, ply_root, 'train', transform=_modelnet_tfs, sub_cat=None, number_of_view=1, data_augment=True)
    mv_ds_test = MultiViewDataSet(root, ply_root, 'test', transform=_modelnet_tfs, sub_cat=None, number_of_view=1)

    print('Avaiable Classes are:')
    print(mv_ds.class_to_idx)

    ds_loader = DataLoader(mv_ds, batch_size=opt.batch_size, drop_last=True, shuffle=True)
    ds_loader_test = DataLoader(mv_ds_test, batch_size=opt.batch_size, shuffle=True)
    
    # Initialize the model & optimizer
    model = PCEncoder(core='pointnet')
    classifier = AuxClassifier(1024, 40)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=opt.lr,
        betas=(.9, .999)
    )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay, gamma=.5)

    model.cuda()
    classifier.cuda()

    model.train()
    for epoch in range(1, 101):
        # Track running performance
        running_loss = 0.
        running_acc = 0.

        for _, pcs, _, label in tqdm(ds_loader, desc='Epoch {:d} training'.format(epoch)):
            pcs = Variable(pcs.transpose(2, 1).cuda())
            label = Variable(label.cuda())

            optimizer.zero_grad()
            pc_feat = model(pcs)
            pred = classifier(pc_feat)

            pred_label = pred.data.max(1)[1]

            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += pred_label.eq(label.data).cpu().sum()
            # print(running_acc.item())
        print(f'Running CrossEntropy is {running_loss / len(mv_ds)}, Running Acc is {running_acc.item() / len(mv_ds)} at Epoch {epoch}')

        if epoch % opt.val_interval == 0:
            # EVALUATION CODE -- TEST Mode
            model.eval()
            tmp_loss = 0.
            ttl_acc = 0.
            with torch.no_grad():
                for _, pcs, _, label in tqdm(ds_loader_test, desc='Epoch {:d} training'.format(epoch)):
                    pcs = Variable(pcs.transpose(2, 1).cuda())
                    label = Variable(label.cuda())

                    pc_feat = model(pcs)
                    pred = classifier(pc_feat)

                    pred_label = pred.data.max(1)[1]

                    loss = criterion(pred, label)

                    tmp_loss += loss.item()
                    ttl_acc += pred_label.eq(label).cpu().sum()
            print(f'Test CrossEntropy is {tmp_loss / len(mv_ds_test)}, Test Accuracy is {ttl_acc.item() / len(mv_ds_test)} at Epoch {epoch}')
            model.train()
            pass

        if epoch & opt.save_interval == 0 or epoch == opt.epoch:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'pretrained_pcencoder_{opt.core}.pt'))

        scheduler.step()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Paths
    parser.add_argument('--root', type=str, required=True, help="Path to the image")
    parser.add_argument('--proot', type=str, required=True, help="Path to the ply")

    # Parameters for training:
    parser.add_argument('--epoch', type=int ,default=250, help='Number of epochs to training (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
    parser.add_argument('--lr_decay', type=float, default=40, help='Decay learning rate every LR_DECAY epoches')
    parser.add_argument('--core', type=str, default='pointnet', help='The core of the PCEncoder')

    # Experiment parameters: EXP_NAME, checkpoint path, etc.
    parser.add_argument('--name', type=str, default='pretrain_pointnet', help='Experiment Name')
    parser.add_argument('--dir_name', type=str, default='', help='Name of the log folder')
    parser.add_argument('--model_path', type=str, default='/home/yulin/Few_shot_point_cloud_reconstruction/checkpoint')
    parser.add_argument('--save_interval', type=int, default=20, help='Save Interval')
    parser.add_argument('--val_interval', type=int, default=10, help='Test Interval')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch_size')

    conf = parser.parse_args()
    main(conf)
