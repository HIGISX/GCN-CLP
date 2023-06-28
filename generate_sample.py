import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from pathlib import Path
import os
import pickle
import argparse
import numpy as np
from logzero import logger
import networkx as nx
from mclp_generator import MCLPGenerator
from lscp_generator import LSCPGenerator

def generate_graph(input_file,output_path,problem):
    data = np.load(input_file)
    prefix = "station_mclp"
    n = np.size(data["arr_0"], 0)
    costs = data["arr_1"].astype(int)
    C = data["arr_2"].astype(int)
    label = data["arr_3"].astype(int)
    #mclp
    if problem == "mclp":
        cover_range = data["arr_4"].astype(int)
    edge_list = []
    weighted = True
    for i in range(n):
        index_array = np.where(C[:, i] == 1)
        for j in index_array[0]:
            if j > i:
                edge_list.append((i, j))
    G = nx.empty_graph(n)
    if problem == "mclp":
        G.graph["range"]=cover_range
    G.add_edges_from(edge_list)
    if weighted:
        weight_mapping = {vertex: int(weight) for vertex, weight in
                          zip(G.nodes, costs)}
        nx.set_node_attributes(G, values=weight_mapping, name='weight')
    # add two feature:
    cover_num_mapping = {
        vertex: np.sum(cover) for vertex, cover in
        zip(G.nodes,C)
    }
    nx.set_node_attributes(G, values=cover_num_mapping, name='cover_num')
    cover_mapping = {
        vertex: covers for vertex, covers in
        zip(G.nodes,C)
    }
    nx.set_node_attributes(G, values=cover_mapping, name='cover')
    label_mapping = {vertex: int(label[vertex]) for vertex in G.nodes}
    nx.set_node_attributes(G, values=label_mapping, name='label')

    output_file = output_path + '/' + f"{prefix}.gpickle"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    with open(output_file, "wb") as f:
         pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
    cost_total = np.dot(costs,label.T)
    return label,cost_total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate files for train and test.")
    parser.add_argument("problem", type=str, help="problem type.", choices=["mclp", "lscp"])
    parser.add_argument("input_folder", type=str, action="store", help="Directory containing input files.")
    parser.add_argument("output_path", type=str, action="store",  help="Path of output files")

    args = parser.parse_args()
    file_list = sorted(os.listdir(args.input_folder))
    print(file_list)
    for file in file_list:
        prefix = file.split(".")[0]
        output_file = args.output_path + '/' + f"{prefix}.gpickle"
        if os.path.exists(output_file):
            continue
        label,cost_total = generate_graph(os.path.join(args.input_folder,file),args.output_path,args.problem)
        label_num = int(sum(label))
        print(file)
    if args.problem=="lscp":
        generator = LSCPGenerator()
    else:
        generator = MCLPGenerator()
    generator.process(Path(args.output_path))

