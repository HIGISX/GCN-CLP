import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv
import math
import numpy as np
from post_process import lscp_greedy, mclp_greedy
from torch.distributions.categorical import Categorical
def make_heads(qkv, n_heads):
    shp = (qkv.size(0),qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)
class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
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
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, heads=[8, 1]
    ):
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
            if i == len(self.gat_layers) - 1:  # last layer
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
            greedy_upper_bounds, label_coverd = mclp_greedy(
                output, probmaps, A, labels.tolist()
            )
            gap = abs(label_coverd - greedy_upper_bounds)
        output = output.permute(1, 0)
        weight = torch.sum(labels).item() / output.shape[1]
        self.ce_func = nn.BCELoss(
            torch.tensor([1 / weight], device=torch.device("cuda:0")),
            reduction="none",
        )
        loss = torch.min(torch.mean(self.ce_func(output, _labels), axis=1))
        loss = loss + gap
        return loss, greedy_upper_bounds

class DecoderAttention(torch.nn.Module):
    def __init__(
            self,
            embedding_dim,
            n_heads=8,
            tanh_clipping=50.0,
            multi_pointer=8,
            multi_pointer_level=3,
            add_more_query=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping

        self.Wq_graph = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wq_first = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.Wq_last = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wq = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)


        self.Wk = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.q_graph = None  # saved q1, for multi-head attention
        self.q_first = None  # saved q2, for multi-head attention
        self.glimpse_k = None  # saved key, for multi-head attention
        self.glimpse_v = None  # saved value, for multi-head_attention
        self.logit_k = None  # saved, for single-head attention
        self.group_ninf_mask = None  # reference to ninf_mask owned by state
        self.multi_pointer = multi_pointer  #
        self.multi_pointer_level = multi_pointer_level
        self.add_more_query = add_more_query

    def reset(self, matrix, embeddings, G, trainging=True):
        B, N, H = embeddings.shape

        self.range_matrix = matrix  # [:,:2]
        self.embeddings = embeddings
        self.embeddings_group = self.embeddings.unsqueeze(1).expand(B, G, N, H)
        graph_embedding = self.embeddings.mean(dim=1, keepdim=True)

        self.q_graph = self.Wq_graph(graph_embedding)
        self.q_first = None
        self.logit_k = embeddings.transpose(1, 2)
        if self.multi_pointer > 1:
            self.logit_k = make_heads(
                self.Wk(embeddings), self.multi_pointer
            ).transpose(
                2, 3
            )

    def forward(self, last_node, group_ninf_mask, last_coverd):
        B, N, H = self.embeddings.shape
        G = group_ninf_mask.size(1)

        # Get last node embedding
        last_node_index = last_node.view(B, G, 1).expand(-1, -1, H)
        last_node_embedding = self.embeddings.gather(1, last_node_index)
        q_last = self.Wq_last(last_node_embedding)

        # Get first node embedding
        if self.q_first is None:
            self.q_first = self.Wq_first(last_node_embedding)
        group_ninf_mask = group_ninf_mask.detach()

        mask_visited = group_ninf_mask.clone()
        mask_visited[mask_visited == -np.inf] = 1.0
        mask_visited[mask_visited == np.inf] = 0.0

        last_node_range = last_coverd

        intersection = torch.matmul(last_node_range,self.range_matrix)
        union = torch.sum(torch.logical_or(last_node_range.unsqueeze(3), self.range_matrix.unsqueeze(1)), dim=-1)
        iou = intersection / union

        if self.add_more_query:
            final_q = q_last + self.q_first + self.q_graph
        else:
            final_q = q_last + self.q_first + self.q_graph

        if self.multi_pointer > 1:
            final_q = make_heads(
                self.wq(final_q), self.n_heads
            )
            score = (torch.matmul(final_q, self.logit_k) / math.sqrt(H)) - (
                iou
            ).unsqueeze(
                1
            )  # (B,n_head,G,N)
            if self.multi_pointer_level == 1:
                assert not torch.any(torch.isnan(score))
                assert not torch.any(torch.isnan(torch.tanh(score.mean(1))))
                score_clipped = self.tanh_clipping * torch.tanh(score.mean(1))
            elif self.multi_pointer_level == 2:
                score_clipped = (self.tanh_clipping * torch.tanh(score)).mean(1)
            else:
                # add mask
                score_clipped = self.tanh_clipping * torch.tanh(score)
                mask_prob = group_ninf_mask.detach().clone()
                mask_prob[mask_prob == -np.inf] = -1e8
                mask_prob[mask_prob == np.inf] = 1e8

                score_masked = score_clipped + mask_prob.unsqueeze(1)
                probs = F.softmax(score_masked, dim=-1).mean(1)
                return probs
        else:
            score = torch.matmul(final_q, self.logit_k) / math.sqrt(
                H
            )
            score_clipped = self.tanh_clipping * torch.tanh(score)

        # add mask
        mask_prob = group_ninf_mask.detach().clone()
        mask_prob[mask_prob == -np.inf] = -1e8
        mask_prob[mask_prob == np.inf] = 1e8
        score_masked = score_clipped + mask_prob

        probs = F.softmax(score_masked, dim=2)

        return probs

class FrameModel(torch.nn.Module):
    def __init__(self, in_feats, prob_maps):
        super().__init__()
        self.GCN = GCN(
            in_feats,  # input
            128,  # dimensions in hidden layers
            prob_maps,  # probability maps
            3,  # hidden layers
            F.relu,
            0.3,
        ).cuda()
        self.liner = torch.nn.Linear(in_feats,128).cuda()
        self.decoder = DecoderAttention(
            embedding_dim=128,
            n_heads=8,
            tanh_clipping=50,
            multi_pointer=8,
            multi_pointer_level=1,
            add_more_query=True,
        ).cuda()

    def forward(self,graph,features,prob_maps,A,env,s,group_idx_range,batch_idx_range,training=True,random=False):
        d = False
        device = 'cuda:0'
        if random:
            embeddings = self.liner(torch.tensor(features,dtype=torch.float32))
        else:
            embeddings = self.GCN(graph, features)
            embeddings = torch.tensor(embeddings)

        log_prob = torch.zeros(1, prob_maps, device=device)
        self.decoder.reset(torch.tensor(A,dtype=torch.float32).to(device).unsqueeze(0), embeddings.unsqueeze(0),prob_maps, trainging=False)
        if training == False:
            first_action = torch.randperm(prob_maps)[None, :prob_maps].expand(1, prob_maps).to(device)
            pi = first_action[..., None]
            s, r, d = env.step(first_action)
            self.decoder.reset(torch.tensor(A,dtype=torch.float32).to(device).unsqueeze(0), embeddings.unsqueeze(0), prob_maps,trainging=False)
            while not d:
                tensor = torch.sum(s.current_covered, dim=2) > 0
                last_coverd = tensor.float()
                action_probs = self.decoder(
                    s.current_node.to(device), s.ninf_mask.to(device), last_coverd
                )
                action = action_probs.argmax(dim=2)
                pi = torch.cat([pi, action[..., None]], dim=-1)
                s, r, d = env.step(action)
            return s,r,d
        while not d:
            if s.current_node is None:
                # pick start node randomly and multi cases
                first_action = torch.randperm(prob_maps)[None, :prob_maps].expand(1, prob_maps)
                s, r, d = env.step(first_action.to(device))
                continue

            last_node = s.current_node
            tensor = torch.sum(s.current_covered, dim=2) > 0
            last_coverd = tensor.float()
            action_probs = self.decoder(
                last_node.to(device), s.ninf_mask.to(device), last_coverd
            )

            m = Categorical(action_probs.reshape(1 * prob_maps, -1))
            action = m.sample().view(1, prob_maps)
            chosen_action_prob = (
                    action_probs[batch_idx_range,group_idx_range, action].reshape(1,prob_maps)
                    + 1e-8
            )
            log_prob += chosen_action_prob.log()
            s, r, d = env.step(action)

        return s,r,d,log_prob