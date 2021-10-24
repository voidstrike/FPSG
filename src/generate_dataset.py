# import sys
import os
import argparse

from datasets.mv_dataset import synsetid_to_cate


_SHAPENET_ID2NAME = synsetid_to_cate

_SHAPENET_NAME2ID = {_SHAPENET_ID2NAME[eachKey]: eachKey for eachKey in _SHAPENET_ID2NAME.keys()}


# Modify this dictionary for specific category setting (base classes)
TRAIN_SET_DIC = {
#    'modelnet': ['airplane', 'bed', 'bookshelf', 'chair', 'desk', 'dresser', 'glass_box', 'monitor', 'person', 'plant', 'range_hood',
#    'sofa', 'stool', 'tent', 'tv_stand', 'wardrobe', 'bathtub', 'bench', 'bottle', 'car', 'cone', 'curtain', 'flower_pot', 'guitar', 'lamp', 'mantel',
#    'night_stand', 'piano', 'radio', 'sink', 'stairs', 'table', 'toilet', 'vase', 'xbox'],
     'modelnet': ['airplane', 'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'sofa', 'table', 'toilet'],
    'shapenet': ['airplane', 'camera', 'car', 'clock', 'chair', 'faucet', 'printer', 'rocket']
}

# Modify this dictionary for specific category setting (novel classes)
TEST_SET_DIC = {
#     'modelnet': ['cup', 'keyboard', 'tent', 'guitar', 'door', 'xbox', 'stool', 'bowl', 'radio', 'lamp']
    'modelnet': ['cup', 'keyboard', 'door', 'laptop', 'bowl'],
    # 'modelnet': ['cup', 'keyboard']
    'shapenet': ['bowl', 'cellphone', 'jar', 'mug', 'monitor']
}

def write2file(path, img_corpus, pc_corpus, shapenet=False):
    '''
    Helper function that generates TRAIN/TEST config files
    '''
    ending = '\n'
    if not shapenet:
        with open(path, 'w') as f:
            for idx, (img_path, pc_path) in enumerate(zip(img_corpus, pc_corpus)):
                if idx == len(img_corpus) - 1:
                    ending = ''
                f.write(img_path + '\t' + pc_path + ending)
    else:
        with open(path, 'w') as f:
            for idx, pc_path in enumerate(pc_corpus):
                if idx == len(pc_corpus) - 1:
                    ending = ''
                f.write(pc_path + ending)

def main(opt):
    '''
    Create support file for each class and generate configuration file for TRAIN & TEST 
    '''
    img_root = opt.img_path
    pc_root = opt.pc_path
    dataset = opt.dataset

    # if dataset == 'modelnet' -> Image dir format: root / <label> / <train/test> / <item> / <view>.png
    train_imgs, test_imgs = list(), list()
    train_pcs, test_pcs = list(), list()
    tmp_imgs, tmp_pcs = list(), list()
    
    train_file_path = opt.output + f'{dataset}_train.txt'
    test_file_path = opt.output + f'{dataset}_test.txt'

    if dataset == 'modelnet':
        for label in os.listdir(img_root):
            tmp_imgs, tmp_pcs = list(), list()
            for data_split in ['train', 'test']:
                c_path = os.path.join(img_root, label, data_split)
                ply_path = os.path.join(pc_root, label, data_split)

                for item in os.listdir(c_path):
                    cc_path = os.path.join(c_path, item)
                    ply_item_path = os.path.join(ply_path, f'{item}.ply')
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

            # NOTE: Create folder "modelnet_files" if it's not exist
            if label in TEST_SET_DIC[dataset] or label in TRAIN_SET_DIC[dataset]:
                classes_file = opt.output + f'modelnet_files/{dataset}+{label}.txt'
                write2file(classes_file, tmp_imgs, tmp_pcs)

        write2file(train_file_path, train_imgs, train_pcs)
        write2file(test_file_path, test_imgs, test_pcs)
    else:
        _shape_train = [_SHAPENET_NAME2ID[idx] for idx in TRAIN_SET_DIC[dataset]]
        _shape_test = [_SHAPENET_NAME2ID[idx] for idx in TEST_SET_DIC[dataset]]
        for label in _SHAPENET_ID2NAME.keys():
            tmp_items = list()
            for data_split in ['train', 'test']:
                file_path = os.path.join(pc_root, f'{label}_{data_split}.txt')
                item_root = os.path.join(pc_root, label)

                if label in _shape_train:
                    with open(file_path, 'r') as f:
                        for eachLine in f.readlines():
                            filename=eachLine.strip()
                            item_path = os.path.join(os.path.join(item_root, filename), 'models')
                            train_pcs.append(item_path)
                            tmp_items.append(item_path)
                if label in _shape_test:
                    with open(file_path, 'r') as f:
                        for eachLine in f.readlines():
                            filename=eachLine.strip()
                            item_path = os.path.join(os.path.join(item_root, filename), 'models')
                            test_pcs.append(item_path)
                            tmp_items.append(item_path)

            # NOTE: Create folder "shapenet_files" if it's not exist
            classes_file = opt.output + f'shapenet_files/{dataset}+{label}.txt'
            write2file(classes_file, None, tmp_items, shapenet=True)

        write2file(train_file_path, None, train_pcs, True)
        write2file(test_file_path, None, test_pcs, True)

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path', type=str, required=True, help='Path to the image directory;')
    parser.add_argument('--pc_path', type=str, required=True, help='Path to the pc directory [Could be any string if dataset is shapenet];')
    parser.add_argument('--dataset', type=str, required=True, choices=['modelnet', 'shapenet'], help='Type of the dataset;')
    parser.add_argument('--output', type=str, default='./', help='Root path of the test_split [default: ./];')

    conf = parser.parse_args()
    main(conf)
