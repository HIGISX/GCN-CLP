import os.path
import pathlib
import dgl
import networkx as nx
import numpy as np
import random as rd
import json
from tqdm import tqdm
from pathlib import Path
from logzero import logger


class MCLPGenerator:
    def __init__(self):
        self.model_state_path = None

    def load_weights(self, model_state_path):
        self.model_state_path = model_state_path

    def __str__(self):
        return "mclp_generator"

    def directory(self):
        return pathlib.Path(__file__).parent / "mclp_generator"

    def _prepare_instance(source_instance_file, cache_directory, **kwargs):
        raise NotImplementedError("This function should never be called.")

    @staticmethod
    def __prepare_graph(nx_graph, weighted=False):
        labels_given = (
            "label" in rd.sample(nx_graph.nodes(data=True), 1)[0][1].keys()
        )  # sample random node and check if we have a label
        node_attrs = []

        if labels_given:
            node_attrs.append("label")

        if weighted:
            node_attrs.append("weight")
        node_attrs.append("cover")
        node_attrs.append("cover_num")
        g = dgl.from_networkx(nx_graph, node_attrs=node_attrs)
        covering_range = nx_graph.graph["range"]
        if labels_given:
            g.ndata["label"] = g.ndata["label"].to(dgl.backend.data_type_dict["int8"])

        if not weighted:
            g.ndata["weight"] = dgl.backend.tensor(
                np.ones(shape=(g.num_nodes(), 1)),
                dtype=dgl.backend.data_type_dict["int8"],
            )

        # force shape of weights (n x 1 shape)
        g.ndata["weight"] = (
            g.ndata["weight"].reshape(-1, 1).to(dgl.backend.data_type_dict["int8"])
        )
        g.ndata["cover"] = g.ndata["cover"].to(dgl.backend.data_type_dict["int8"])
        g.ndata["cover_num"] = (
            g.ndata["cover_num"].reshape(-1, 1).to(dgl.backend.data_type_dict["int8"])
        )

        return g, covering_range

    def _prepare_instances(
        C, instance_directory: pathlib.Path, cache_directory, **kwargs
    ):
        cache_directory.mkdir(parents=True, exist_ok=True)
        weighted = kwargs.get("weighted", False)
        dest_graphs_file = (
            cache_directory / f"graphs_{'weighted' if weighted else 'unweighted'}.dgl"
        )
        name_mapping_file = cache_directory / f"graph_names.json"
        range_mapping_file = cache_directory / f"graph_range.json"
        last_updated = 0
        if os.path.exists(str(dest_graphs_file)):
            logger.info(f"Found existing graphs file {dest_graphs_file}")
            last_updated = os.path.getmtime(dest_graphs_file)
            gs, _ = dgl.load_graphs(str(dest_graphs_file))

        for graph_path in instance_directory.rglob("*.gpickle"):
            graph_file = graph_path.resolve()
            source_mtime = os.path.getmtime(graph_file)
            if source_mtime > last_updated or not os.path.exists(name_mapping_file):
                logger.info(
                    f"Updated graph file: {graph_file} (or name mapping was not existing yet)."
                )
                logger.info(f"Re-converting all graphs... this can take a while.")
                gs = []
                graph_names = []
                graph_covering_range = []
                for graph_path in tqdm(instance_directory.rglob("*.gpickle")):
                    graph_file = graph_path.resolve()
                    nx_graph = nx.read_gpickle(graph_file)
                    g, covering_range = C.__prepare_graph(nx_graph, weighted=weighted)
                    gs.append(g)
                    graph_covering_range.append(covering_range.item())
                    graph_names.append(
                        os.path.splitext(os.path.basename(graph_file))[0]
                    )
                dgl.save_graphs(str(dest_graphs_file), gs)
                with open(name_mapping_file, "w", encoding="utf-8") as f:
                    json.dump(
                        graph_names, f, ensure_ascii=False, sort_keys=True, indent=4
                    )
                with open(range_mapping_file, "w", encoding="utf-8") as f:
                    json.dump(
                        graph_covering_range,
                        f,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=4,
                    )
                break

    def process(self, train_data_path: pathlib.Path):
        cache_directory = train_data_path / "preprocessed" / str(self)
        weighted = True
        self._prepare_instances(train_data_path, cache_directory, weighted=weighted)

        logger.info("Invoking training of " + str(self))


if __name__ == "__main__":
    generator = MCLPGenerator()
    generator.process(Path("D:/dataset/1000-mclp-train"))
