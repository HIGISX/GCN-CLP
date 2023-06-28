import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv,GATConv
from post_process import lscp_greedy, mclp_greedy

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self._LR_multiplier_list = []
        self._LR_rate = None
        self._violations_epoch = []
        self.const_avg_batch = 0

        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, activation=torch.sigmoid))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class GAT(nn.Module):
    def __init__(self,
                     in_feats,
                     n_hidden,
                     n_classes,
                     n_layers,
                     activation,
                     heads=[8,1]):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self._violations_epoch = []
        self.const_avg_batch = 0
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(
                in_feats,
                n_hidden,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=activation,
            )
        )
        for i in range(n_layers - 2):
            self.gat_layers.append(
                GATConv(
                    n_hidden * heads[0],
                    n_hidden,
                    heads[0],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=activation,
                )
            )
        self.gat_layers.append(
            GATConv(
                n_hidden * heads[0],
                n_classes,
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=torch.sigmoid,
                )
        )

    def forward(self, g, inputs):
        h = inputs.to(torch.float32)
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == len(self.gat_layers)-1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
        return h

class HindsightLoss(nn.Module):
    def __init__(self):
        super(HindsightLoss, self).__init__()
        self.lscp_greedy = lscp_greedy
        self.mclp_greedy = mclp_greedy
        self.ce_func = None

    def forward(self, output, labels, A, problem):
        # output 进行后处理得到上界
        probmaps = output.shape[1]
        _labels = torch.unsqueeze(labels, 0)
        _labels = _labels.float().repeat(probmaps, 1)
        if problem == "lscp":
            greedy_upper_bounds = self.lscp_greedy(output, probmaps, A)
            y = torch.sum(labels).item()
            gap = abs(y - greedy_upper_bounds)
        else:
            greedy_upper_bounds, label_coverd = mclp_greedy(output, probmaps, A, labels.tolist())
            gap = abs(label_coverd - greedy_upper_bounds)
        output = output.permute(1, 0)
        weight = torch.sum(labels).item()/output.shape[1]
        self.ce_func = nn.BCELoss(torch.tensor([1/weight,1-1/weight], device=torch.device("cuda:0")), reduction="none")
        loss = torch.min(torch.mean(self.ce_func(output, _labels), axis=1))
        loss = loss + gap
        return loss, greedy_upper_bounds
