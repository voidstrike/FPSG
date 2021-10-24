import torch

# Episodic Extraction
def extract_episode(n_support, n_query, d):
    # img_data : N x C x H x W
    # pc_data : N x 3 x 2048 (current hardcoded)
    n_examples = d['img_data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_idx = torch.randperm(n_examples)[:(n_support + n_query)]
    support_idx = example_idx[:n_support]
    query_idx = example_idx[n_support:]

    imgs = d['img_data'][support_idx]
    imgq = d['img_data'][query_idx]
    pcs = d['pc_data'][support_idx]
    pcq = d['pc_data'][query_idx]

    return {
        'class': d['class'],
        'xs': imgs,
        'xq': imgq,
        'pcs': pcs,
        'pcq': pcq,
        'tmp': query_idx.item()
    }
    

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self,):
        return self.n_episodes

    def __iter__(self,):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self,):
        return self.n_classes

    def __iter__(self,):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])


class SequentialBatchSamplerV2(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self,):
        return self.n_classes

    def __iter__(self,):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])