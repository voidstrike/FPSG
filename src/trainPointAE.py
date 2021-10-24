import argparse
import torch
import os
import time

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torchvision.transforms as tfs

from datasets.mv_dataset import MultiViewDataSet, ShapeNet55
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.point_cloud_net import PCEncoder
from models.support_models import AuxClassifier
from tqdm import tqdm

# Auxiliary training script to get a pretrained point cloud encoder

# Predefined Macro for ShapeNet <- modify HERE, category_configuration must be generated first (set to None if you want to use all categories)
_SHAPE_CAT = ['02691156', '02942699', '02958343', '03046257', '03001627', '03325088', '04004475', '04099429']
# Predefined Macro for ModelNet <- modify HERE (set to None if you want to use all categories)
_MODEL_CAT = ['airplane', 'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'sofa', 'table', 'toilet']

# Image Transformation functions -- USELESS here
def _build_tfs(model_name):
    ans = tfs.Compose([
        tfs.CenterCrop(224 if model_name=='modelnet' else 256),
        tfs.Resize(224),
        tfs.ToTensor(),
        tfs.Normalize((.5, .5, .5), (.5, .5, .5))
    ])
    return ans

_MODEL_TFS, _SHAPE_TFS = _build_tfs('modelnet'), _build_tfs('shapenet')

def main(opt):
    # Get dataset path
    root, ply_root = opt.root, opt.proot
    checkpoint_path = os.path.join(opt.model_path, opt.name)

    # Create dataset objects
    if opt.dataset == 'modelnet':
        mv_ds = MultiViewDataSet(root, ply_root, 'train', transform=_MODEL_TFS, sub_cat=_MODEL_CAT, number_of_view=1, data_augment=False)
        mv_ds_test = MultiViewDataSet(root, ply_root, 'test', transform=_MODEL_TFS, sub_cat=_MODEL_CAT, number_of_view=1)
        num_cat = len(_MODEL_CAT)
    elif opt.dataset == 'shapenet':
        mv_ds = ShapeNet55(root, _SHAPE_CAT, 'train', transform=_SHAPE_TFS, data_augment=False)
        mv_ds_test = ShapeNet55(root, _SHAPE_CAT, 'test', transform=_SHAPE_TFS)
        num_cat = len(_SHAPE_CAT)

    print('Avaiable Classes are:')
    print(mv_ds.class_to_idx)

    ds_loader = DataLoader(mv_ds, batch_size=opt.batch_size, drop_last=True, shuffle=True)
    ds_loader_test = DataLoader(mv_ds_test, batch_size=opt.batch_size, shuffle=True)
    
    # Initialize model & optimizer
    model = PCEncoder(core='pointnet')

    classifier = AuxClassifier(1024, num_cat)
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=opt.lr,
        betas=(.9, .999)
    )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay, gamma=.5)

    model.cuda()
    classifier.cuda()

    # Actual training procedure
    model.train()
    for epoch in range(1, 151):
        # Performance trackers
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

        print(f'Running CrossEntropy is {running_loss / len(mv_ds)}, Running Acc is {running_acc.item() / len(mv_ds)} at Epoch {epoch}')

        # Evaluation part
        if epoch % opt.val_interval == 0:
            model.eval()
            tmp_loss = 0.
            ttl_acc = 0.
            with torch.no_grad():
                for _, pcs, _, label in tqdm(ds_loader_test, desc='Epoch {:d} evaluating'.format(epoch)):
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
            # NOTE: the trained model will be saved with the following name
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'{opt.name}_{opt.core}.pt'))

        scheduler.step()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data Paths
    parser.add_argument('--root', type=str, required=True, help="Path to the image dir;")
    parser.add_argument('--proot', type=str, required=True, help="Path to the PLY dir (arbitary value for ShapeNet);")
    parser.add_argument('--dataset', type=str, required=True, choices=['modelnet', 'shapenet'], help='Type of the dataset;')

    # Training Parameters
    parser.add_argument('--epoch', type=int ,default=150, help='Number of epochs to training [default: 150];')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate [default: 1e-3];')
    parser.add_argument('--lr_decay', type=float, default=40, help='Decay learning rate every LR_DECAY epoches [default: 40];')
    parser.add_argument('--core', type=str, default='pointnet', choices=['pointnet', 'dgcnn'], help='The core of the PCEncoder [default: pointnet];')

    # Experiment Parameters: EXP_NAME, checkpoint path, etc.
    parser.add_argument('--name', type=str, default='pretrain_pointnet', help='Experiment Name [default: pretrain_pointnet];')
    parser.add_argument('--model_path', type=str, default='../checkpoint', help='Path to the check point folder [default: ../checkpoint/];')
    parser.add_argument('--save_interval', type=int, default=20, help='Number of epochs between each save [default: 20];')
    parser.add_argument('--val_interval', type=int, default=10, help='Number of epochs between each training validation [default: 10];')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch_size [default: 32];')

    conf = parser.parse_args()
    main(conf)
