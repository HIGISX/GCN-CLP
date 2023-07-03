import dgl
import time
import numpy as np
from logzero import logger
import torch
from tqdm import tqdm
from utils import _load_model
from post_process import mclp_greedy, lscp_greedy


class Solver:
    def __init__(self):
        self.model_prob_maps = 32
        self.weight_file = None
        self.num_nodes = 50
        self.cuda_dev = torch.device("cuda:0")
        self.noise_as_prob_maps = False

    def load_model(self):
        self.model = _load_model(
            self.num_nodes + 1,
            self.model_prob_maps,
            weight_file=self.weight_file,
            cuda_dev=self.cuda_dev,
        )
        self.model.eval()

    def infer_prob_maps(self, graph):
        features = torch.cat((graph.ndata["cover"], graph.ndata["cover_num"]), dim=1)
        if not self.noise_as_prob_maps:
            out = self.model(graph, features)
        else:
            # Replace GNN output probability maps with random noise
            out = torch.rand(features.shape[0], self.max_prob_maps)
        return out


def solve(args):
    cuda = bool(args.cuda_devices)
    if not args.pretrained_weights:
        raise ValueError("--pretrained_weights flag is required for solving! Exiting.")
    dgl_graphs = dgl.load_graphs(
        str(args.input / f"graphs_{'weighted' if args.weighted else 'unweighted'}.dgl")
    )[0]
    s = Solver()
    node_num = dgl_graphs[0].number_of_nodes()
    s.num_nodes = node_num
    s.model_prob_maps = args.model_prob_maps
    s.weight_file = args.pretrained_weights

    s.noise_as_prob_maps = args.noise_as_prob_maps
    opt_gaps = []
    validate_graphs = []
    ys = []
    start_time = time.time()
    s.load_model()
    for idx, g in enumerate(tqdm(dgl_graphs)):
        if cuda:
            g = g.to(torch.device("cuda:0"))
        if args.self_loops:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
        else:
            g = dgl.remove_self_loop(g)
        validate_graphs.append(g)
        prob_maps = s.infer_prob_maps(g)
        labels = g.ndata["label"]
        num_prob_maps = prob_maps.shape[1]
        A = g.adj(transpose=True, scipy_fmt="coo").toarray()
        if args.problem == "mclp":
            y_pre_covering, y_covering = mclp_greedy(
                prob_maps, num_prob_maps, A, labels.tolist()
            )
            opt_gap = abs(y_pre_covering - y_covering) / y_covering * 100
            ys.append(y_pre_covering)
        else:
            y_pre = lscp_greedy(prob_maps, num_prob_maps, A)
            y = torch.sum(labels).item()
            opt_gap = abs(y_pre - y) / y * 100
            ys.append(y_pre)
        opt_gaps.append(opt_gap)
    period_time = time.time() - start_time
    mean_opt_gap = np.mean(opt_gaps)
    mean_ys = np.mean(ys)
    std_opt_gap = np.std(opt_gaps)
    logger.info(
        "Validation optimality gap: {:.3f}% +- {:.3f}".format(mean_opt_gap, std_opt_gap)
    )
    logger.info("y mean: {:.3f}".format(mean_ys))
    logger.info(
        "problem:{},problem scale:{},mean_opt_gap:{},period_time:{}".format(
            args.problem, node_num, mean_opt_gap, period_time
        )
    )
    logger.info("Done with all graphs, exiting.")
