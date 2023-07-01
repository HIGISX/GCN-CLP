import os
import torch
import dgl
import random
import time
import numpy as np
from tqdm import tqdm
from logzero import logger

from post_process import lscp_greedy, mclp_greedy
from model import HindsightLoss
from utils import _load_model


class Train:
    def __init__(self):
        self._violations_epoch = []
        self.const_avg_batch = 0
        self._costs = []
        self.A_list = []

    def validate(self, validate_graphs, model, problem="mclp"):
        # Validate
        if model:
            model.eval()
        print(f"\nValidating on {len(validate_graphs)} ")
        opt_gaps = []
        cuda = True
        ys = []
        for gidx, graph in enumerate(tqdm(validate_graphs)):
            if cuda:
                graph = graph.to(0)
            features = torch.cat(
                (graph.ndata["cover"], graph.ndata["cover_num"]), dim=1
            )
            labels = graph.ndata["label"]
            if model:
                prob_maps = model(graph, features)
            else:
                prob_maps = torch.rand(features.shape[0], 32)
            num_prob_maps = prob_maps.shape[1]
            A = graph.adj(transpose=True, scipy_fmt="coo").toarray()
            if problem == "mclp":
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

        mean_opt_gap = np.mean(opt_gaps)
        mean_ys = np.mean(ys)
        std_opt_gap = np.std(opt_gaps)
        logger.info(
            "Validation optimality gap: {:.3f}% +- {:.3f}".format(
                mean_opt_gap, std_opt_gap
            )
        )
        logger.info("y mean: {:.3f}".format(mean_ys))
        return mean_opt_gap

    def train(self, args):
        self_loop = args.self_loops
        cuda = bool(args.cuda_devices)
        prob_maps = args.model_prob_maps
        problem = args.problem

        if cuda:
            cuda_dev = args.cuda_devices[0]
            if len(args.cuda_devices) > 1:
                logger.warn("More than one cuda device was provided, using " + cuda_dev)
        else:
            cuda_dev = None
        training_graphs = []
        validate_graphs = []
        for file_path in os.listdir(args.input):
            if ".dgl" not in file_path:
                continue
            dgl_graphs = dgl.load_graphs(str(args.input / file_path))[0]
            random.shuffle(dgl_graphs)
            logger.info("Loading training graphs.")
            # for train
            for g in tqdm(dgl_graphs[: len(dgl_graphs) - 800]):
                if self_loop:
                    g = dgl.remove_self_loop(g)
                    g = dgl.add_self_loop(g)
                else:
                    g = dgl.remove_self_loop(g)
                self.node_num = g.number_of_nodes()
                training_graphs.append(g)
                A = g.adj(transpose=True, scipy_fmt="coo").toarray()
                self.A_list.append(A)
            # for validate
            for g in tqdm(dgl_graphs[len(dgl_graphs) - 1000 :]):
                if self_loop:
                    g = dgl.remove_self_loop(g)
                    g = dgl.add_self_loop(g)
                else:
                    g = dgl.remove_self_loop(g)
                self.node_num = g.number_of_nodes()
                validate_graphs.append(g)
            logger.info(f"Loaded {len(dgl_graphs)-1000} graphs for training.")

        model = _load_model(
            self.node_num + 1,
            prob_maps,
            weight_file=args.pretrained_weights,
            cuda_dev=cuda_dev,
        )
        loss_fcn = HindsightLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

        num_epochs = args.epochs
        status_update_every = max(1, int(0.1 * (len(dgl_graphs) - 1000)))
        # initial
        min_gap = 100
        for epoch in range(num_epochs + 1):
            # self.pi = self.pi / 2
            logger.info(f"Epoch {epoch}/{num_epochs}")
            epoch_losses = list()
            # optimizer.param_groups[0]['lr'] = 5e-3
            for gidx, graph in enumerate(tqdm(dgl_graphs[: len(dgl_graphs) - 1000])):
                if self_loop:
                    graph = dgl.remove_self_loop(graph)
                    graph = dgl.add_self_loop(graph)
                else:
                    graph = dgl.remove_self_loop(graph)
                self.node_num = graph.number_of_nodes()
                if self.node_num != 20:
                    continue
                A = graph.adj(transpose=True, scipy_fmt="coo").toarray()
                if cuda:
                    graph = graph.to(cuda_dev)
                features = torch.cat(
                    (graph.ndata["cover"], graph.ndata["cover_num"]), dim=1
                )
                labels = graph.ndata["label"]
                model.train()
                # forward
                output = model(graph, features)
                loss, best_ub = loss_fcn(output, labels, A, problem)
                epoch_losses.append(float(loss))
                # propagate loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if gidx % status_update_every == 0:
                    logger.info(
                        f"Epoch {epoch}/{num_epochs}, Graph {gidx}/{len(training_graphs)}: Average Epoch Loss = {np.mean(epoch_losses)}, Last Training Loss = {loss}"
                    )
            if epoch % 20 == 0:
                torch.save(
                    model.state_dict(),
                    args.output
                    / f"{int(time.time())}_intermediate_model{prob_maps}_{epoch}_{np.mean(epoch_losses):.2f}.torch",
                )
            gap = self.validate(validate_graphs, model, problem=problem)
            if gap < min_gap:
                min_gap = gap
                torch.save(model.state_dict(), args.output / f"best_model_{gap}.torch")
        logger.info(
            f"Final: Average Epoch Loss = {np.mean(epoch_losses)}, Last Training Loss = {loss}"
        )
        torch.save(
            model.state_dict(),
            args.output / f"{int(time.time())}_final_model{prob_maps}.torch",
        )


def scp_test_validate(test_data, weight_file, problem):
    t = Train()
    dgl_graphs = dgl.load_graphs(test_data)[0][:1]
    cuda = torch.cuda.is_available()
    validate_graphs = []
    start_time = time.time()
    for idx, g in enumerate(dgl_graphs):
        if cuda:
            g = g.to(torch.device("cuda:0"))
        if True:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
        validate_graphs.append(g)
    node_num = g.number_of_nodes()
    model = _load_model(
        node_num + 1, 32, weight_file=weight_file, cuda_dev=torch.device("cuda:0")
    )
    mean_opt_gap = t.validate(validate_graphs, model, problem=problem)
    period_time = time.time() - start_time
    print("mean_opt_gap:{},period_time:{}".format(mean_opt_gap, period_time))


if __name__ == "__main__":
    # test
    test_data = (
        "D:/dataset/20-mclp-test-feats/preprocessed/dgl_treesearch/graphs_weighted.dgl"
    )
    weight_file = "D:/models/mclp-20/best_model_1.374878381422499.torch"
    problem = "mclp"
    scp_test_validate(test_data, weight_file, problem)
