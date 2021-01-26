import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class AuxClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_layer=3):
        super(AuxClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_dim)

        self.dropout = nn.Dropout(p=.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self._init_weight()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        # return x # logit

    def _init_weight(self,):
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)


class FCMaskAlloacter(nn.Module):
    def __init__(self, img_dim, proto_dim):
        super(FCMaskAlloacter, self).__init__()
        self.fc1 = nn.Linear(img_dim + proto_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, proto_dim)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.sigmoid(x)
        # return F.softmax(x, dim=1)

    def _init_weight(self,):
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)

class TransMaskAllocater(nn.Module):
    def __init__(self, img_dim, proto_dim, hidden_dim=256):
        super(TransMaskAllocater, self).__init__()
        self.fc_q = nn.Linear(img_dim, hidden_dim)
        self.fc_k = nn.Linear(proto_dim, hidden_dim)
    
    def forward(self, query, key, value):
        # Query -- Img -- 1 (or NP) * 512
        # Key   -- PC  -- 1 (or NP) * 1024
        # Value -- PC  -- 1 (or NP) * 1024
        pass
