import sys
import os

import argparse

TRAIN_SET_DIC = {
    'modelnet': ['airplane', 'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'sofa', 'table', 'toilet']
}

TEST_SET_DIC = {
    'modelnet': ['cup', 'keyboard', 'tent', 'guitar', 'door', 'xbox', 'stool', 'bowl', 'radio', 'lamp']
}

def write2file(path, img_corpus, pc_corpus):
    ending = '\n'
    with open(path, 'w') as f:
        for idx, (img_path, pc_path) in enumerate(zip(img_corpus, pc_corpus)):
            if idx == len(img_corpus) - 1:
                ending = ''
            f.write(img_path + '\t' + pc_path + ending)


def main(opt):
    img_root = opt.img_path
    pc_root = opt.pc_path
    dataset = opt.dataset

    # if dataset == 'modelnet' -> Image dir format: root / <label> / <train/test> / <item> / <view>.png
    train_imgs, test_imgs = list(), list()
    train_pcs, test_pcs = list(), list()
    tmp_imgs, tmp_pcs = list(), list()

    for label in os.listdir(img_root):
        tmp_imgs, tmp_pcs = list(), list()
        for data_split in ['train', 'test']:
            c_path = os.path.join(img_root, label, data_split)
            ply_path = os.path.join(pc_root, label, data_split)

            for item in os.listdir(c_path):
                cc_path = os.path.join(c_path, item)
                ply_item_path = os.path.join(ply_path, item.replace('.off', '.ply'))
                views = list()
                for view in os.listdir(cc_path):
                    views.append(os.path.join(cc_path, view))

                if len(views) != 0:
                    tmp_imgs.append(views[0])
                    tmp_pcs.append(ply_item_path)

                    if label in TEST_SET_DIC[dataset]:
                        test_imgs.append(views[0])
                        test_pcs.append(ply_item_path)
                    elif label in TRAIN_SET_DIC[dataset]:
                        train_imgs.append(views[0])
                        train_pcs.append(ply_item_path)
                    else:
                        pass
                    
        classes_file = opt.output + f'extra_files/{dataset}+{label}.txt'

        write2file(classes_file, tmp_imgs, tmp_pcs)
    

    train_file_path = opt.output + f'{dataset}_train.txt'
    test_file_path = opt.output + f'{dataset}_test.txt'

    write2file(train_file_path, train_imgs, train_pcs)
    write2file(test_file_path, test_imgs, test_pcs)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', type=str, required=True, help='Path to the image directory')
    parser.add_argument('--pc_path', type=str, required=True, help='Path to the pc directory')
    parser.add_argument('--dataset', type=str, required=True, choices=['modelnet', 'shapenet'], help='Type of the dataset')
    parser.add_argument('--output', type=str, default='./', help='Root path of the test_split')

    conf = parser.parse_args()
    main(conf)
