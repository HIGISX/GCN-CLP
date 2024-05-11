import os
import torch
import dgl
import random
import time
import numpy as np
from tqdm import tqdm
from logzero import logger
from torch.distributions.categorical import Categorical
from post_process import lscp_greedy, mclp_greedy,lscp_refine
from model import HindsightLoss,DecoderForLarge,FrameModel
from utils import _load_model
from rl_utils import LSCPEnv,MCLPEnv

class Train:
    def __init__(self):
        self._violations_epoch = []
        self.const_avg_batch = 0
        self._costs = []
        self.A_list = []

    def validate(self, validate_graphs, frame_model, problem="mclp"):
        # Validate
        if frame_model:
            frame_model.eval()
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
            A = graph.adj(transpose=True, scipy_fmt="coo").toarray()
            if problem == "lscp":
                env = LSCPEnv(torch.tensor(A,dtype=torch.float32).unsqueeze(0).to("cuda:0"))
            elif problem == "mclp":
                p = args.p
                env = MCLPEnv(torch.tensor(A,dtype=torch.float32).unsqueeze(0).to("cuda:0"),p)
            else:
                assert (
                        problem != "lscp" or problem != "mclp"
                ), " doesn't support this problem"
                # action probs
            s, r, d = env.reset(group_size=args.model_prob_maps)
                # N,32
            self.device = 'cuda:0'

            s, r, d = frame_model(graph,features,args.model_prob_maps,A,env,s,None,None,training=False)

            reward,loc = r.max(dim=-1)

            if problem == "mclp":
                y_covering = mclp_greedy(
                     A, labels.tolist()
                )
                y_pre_covering = args.model_prob_maps+reward.cpu()[0]

                opt_gap = abs(y_pre_covering - y_covering) / y_covering * 100
                ys.append(y_pre_covering)
            else:
                solution = list(s.selected_node_list[0][int(loc.item())])[:int(-reward.cpu()[0])]
                y_pre = lscp_refine(solution,A)
                #y_pre = -reward.cpu()[0]
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
        if not os.path.exists(args.output):
            os.mkdir(args.output)

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
            dgl_graphs = dgl.load_graphs(str(args.input / file_path))[0][:10000]
            random.shuffle(dgl_graphs)
            logger.info("Loading training graphs.")
            # for train
            for g in tqdm(dgl_graphs[: int(0.9*len(dgl_graphs))]):
                if self_loop:
                    g = dgl.remove_self_loop(g)
                    g = dgl.add_self_loop(g)
                else:
                    g = dgl.remove_self_loop(g)

                if g.number_of_nodes()!=prob_maps:
                    print(g.number_of_nodes(),"The node is not match!")
                    continue
                self.node_num = g.number_of_nodes()
                training_graphs.append(g)
                A = g.adj(transpose=True, scipy_fmt="coo").toarray()
                self.A_list.append(A)
            # for validate
            for g in tqdm(dgl_graphs[int(0.9*len(dgl_graphs)):]):
                if self_loop:
                    g = dgl.remove_self_loop(g)
                    g = dgl.add_self_loop(g)
                else:
                    g = dgl.remove_self_loop(g)
                self.node_num = g.number_of_nodes()
                if self.node_num!=prob_maps:
                    print(self.node_num,"The node is not match!")
                    continue
                validate_graphs.append(g)
            logger.info(f"Loaded {len(validate_graphs)} graphs for training.")

        frame_model = FrameModel(
            self.node_num + 1,
            prob_maps
            )

        loss_fcn = HindsightLoss()
        optimizer = torch.optim.Adam(frame_model.parameters(), lr=args.lr, weight_decay=5e-4)

        num_epochs = args.epochs
        status_update_every = max(1, int(0.1 * len(training_graphs)))
        # initial
        min_gap = 100
        batch_idx_range = torch.arange(1)[:, None].expand(1, prob_maps)
        group_idx_range = torch.arange(prob_maps)[None, :].expand(1, prob_maps)
        for epoch in range(num_epochs + 1):

            logger.info(f"Epoch {epoch}/{num_epochs}")
            epoch_losses = list()
            for gidx, graph in enumerate(tqdm(training_graphs)):
                if self_loop:
                    graph = dgl.remove_self_loop(graph)
                    graph = dgl.add_self_loop(graph)
                else:
                    graph = dgl.remove_self_loop(graph)

                self.node_num = graph.number_of_nodes()

                A = graph.adj(transpose=True, scipy_fmt="coo").toarray()
                if cuda:
                    graph = graph.to(cuda_dev)
                features = torch.cat(
                    (graph.ndata["cover"], graph.ndata["cover_num"]), dim=1
                )
                if problem == "lscp":
                    env = LSCPEnv(torch.tensor(A,dtype=torch.float32).unsqueeze(0).to("cuda:0"))
                elif problem == "mclp":
                    env = MCLPEnv(torch.tensor(A,dtype=torch.float32).unsqueeze(0).to("cuda:0"),args.p)
                else:
                    assert (
                            problem != "lscp" or problem != "mclp"
                    ), " doesn't support this problem"
                #labels = graph.ndata["label"]
                # action probs
                s, r, d = env.reset(group_size=prob_maps)
                # N
                self.device = 'cuda:0'

                s, r, d , log_prob= frame_model(graph,features,prob_maps,A,env,s,group_idx_range,batch_idx_range)
                r_trans = r.to(self.device)
                #reward = reward.max(dim=-1).values
                labels = graph.ndata["label"]
                loss = loss_fcn(log_prob,r_trans, labels)
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
                    frame_model.state_dict(),
                    args.output
                    / f"{int(time.time())}_{self.node_num}_intermediate_model{prob_maps}_{epoch}_{np.mean(epoch_losses):.2f}.torch",
                    )
            gap = self.validate(validate_graphs, frame_model, problem=problem)
            if gap < min_gap:
                min_gap = gap
                torch.save(frame_model.state_dict(), args.output / f"best_model_{self.node_num}_{gap}.torch")
        logger.info(
            f"Final: Average Epoch Loss = {np.mean(epoch_losses)}, Last Training Loss = {loss}"
        )
        torch.save(
            frame_model.state_dict(),
            args.output / f"{int(time.time())}_{self.node_num}_final_model{prob_maps}.torch",
            )


def scp_test_validate(test_data, weight_file, problem):
    t = Train()
    dgl_graphs = dgl.load_graphs(test_data)[0]
    cuda = torch.cuda.is_available()
    validate_graphs = []
    start_time = time.time()
    device = torch.device("cuda:0")
    for idx, g in enumerate(dgl_graphs):
        if cuda:
            g = g.to(device)
        if True:
            g = dgl.remove_self_loop(g)
            g = dgl.add_self_loop(g)
        validate_graphs.append(g)
    node_num = g.number_of_nodes()
    frame_model = FrameModel(
        node_num + 1,
        node_num
    )
    model = frame_model.to(device)
    model.load_state_dict(torch.load(weight_file))
    mean_opt_gap = t.validate(validate_graphs, model, problem=problem)
    period_time = time.time() - start_time
    print("mean_opt_gap:{},period_time:{}".format(mean_opt_gap, period_time))

import argparse
from pathlib import Path
if __name__ == '__main__':
    train_flag = True
    # train
    if train_flag:
        t = Train()
        parser = argparse.ArgumentParser(description="Solve SCP based GCN.")
        args = parser.parse_args()
        args.self_loops = True
        args.cuda_devices = [0]
        args.model_prob_maps=50
        #args.problem = "mclp"
        args.problem = "lscp"

        args.output = Path("models/lscp/50")
        args.input = Path("D:/dataset/50-train-feats/preprocessed/dgl_treesearch/")
        args.pretrained_weights=None
        args.lr=0.0001
        args.epochs=200
        args.p = 10
        t.train(args)
    else:
        # test
        parser = argparse.ArgumentParser(description="Solve SCP based GCN.")
        args = parser.parse_args()
        args.p = 10
        args.model_prob_maps=50
        test_data,weight_file, problem = "D:/dataset/50-test-feats/preprocessed/dgl_treesearch/graphs_weighted.dgl","models/lscp/50/best_model_50_1.683085870098254.torch","lscp"
        scp_test_validate(test_data,weight_file,problem)