import argparse
from cogdl.data import Graph
import torch
import numpy as np
from tqdm import tqdm
import os.path as osp
import pickle as pkl

edge_size = 0
node_size = 0

def mask_change(id_mask, node_size):
    mask = torch.zeros(node_size).bool()
    for i in id_mask:
        mask[i] = True
    return mask

def read_node_atts(node_file, pyg_file, label_file=None):
    types = {}
    node_embeds = {}
    count = 0
    node_counts = node_size
    if osp.exists(pyg_file + ".nodes.pyg") == False:
        print("Start loading node information")
        process = tqdm(total=node_counts)
        with open(node_file, 'r') as rf:
            while True:
                line = rf.readline()
                if line is None or len(line) == 0:
                    break
                info = line.strip().split(",")

                node_id = int(info[0])
                node_type = info[1].strip()
                if len(info[2]) < 50:
                    node_embeds[node_id] = np.zeros(256, dtype=np.float32)
                else:
                    node_embeds[node_id] = np.array([x for x in info[2].split(":")], dtype=np.float32)

                types[node_id] = node_type

                count += 1
                if count % 100000 == 0:
                    process.update(100000)

        process.update(node_size % 100000)
        process.close()
        print("Complete loading node information\n")

        print("Num of total nodes:", count)
        print('Node_type Num Num_lack_feature:')

        labels = []
        if label_file is not None:
            labels_info = [x.strip().split(",") for x in open(label_file).readlines()]
            for i in range(len(labels_info)):
                x = labels_info[i]
                item_id = int(x[0])
                label = int(x[1])
                labels.append([item_id, label])

        nodes_dict = {'embeds': node_embeds}
        nodes_dict['labels'] = {}
        nodes_dict['labels'] = labels
        nodes_dict['types'] = {}
        nodes_dict['types'] = types
        print('\n')
        print('Start saving pkl-style node information')
        pkl.dump(nodes_dict, open(pyg_file + ".nodes.pyg", 'wb'), pkl.HIGHEST_PROTOCOL)
        print('Complete saving pkl-style node information\n')

    else:
        nodes = pkl.load(open(pyg_file + ".nodes.pyg", 'rb'))
        node_embeds = nodes['embeds']
        labels = nodes['labels']
        print('label_lens:', len(labels))

    graph = Graph()

    print("Start converting into cogdl graph")
    node_len = 0
    node_len += len(node_embeds)
    graph.x = torch.empty(node_len, 256)
    for nid, embedding in tqdm(node_embeds.items()):
        graph.x[nid] = torch.from_numpy(embedding)

    if label_file is not None:
        graph.y = torch.zeros(node_len, dtype=torch.long) - 1
        for index, label in tqdm(labels, desc="Node labels"):
            graph.y[index] = label

        indices = (graph.y != -1).nonzero().squeeze()
        print("Num of true labeled nodes:{}".format(indices.shape[0]))
        train_val_random = torch.randperm(indices.shape[0])
        train_idx = indices[train_val_random][:int(indices.shape[0] * 0.8)]
        val_idx = indices[train_val_random][int(indices.shape[0] * 0.8):]
        train_mask = mask_change(train_idx, node_len)
        train_mask = train_idx

        valid_mask = mask_change(val_idx, node_len)
        valid_mask = val_idx
        idx = indices[train_val_random][:int(indices.shape[0])]

    print("Complete converting into pyg data\n")

#    print("Start saving into pyg data")
#    torch.save(graph, pyg_file + ".pt")
#    print("Complete saving into pyg data\n")
    return graph,idx


def format_pyg_graph(edge_file="icdm2022_session1_train/icdm2022_session1_edges.csv", node_file="icdm2022_session1_train/icdm2022_session1_nodes.csv", pyg_file="dataset", label_file="icdm2022_session1_train/icdm2022_session1_train_labels.csv"):
    '''
    if osp.exists(pyg_file + ".pt") and args.reload == False:
#        graph = torch.load(pyg_file + ".pt")
        print("PyG graph of " + ("session2" if "session2" in pyg_file else "session1") + " has generated")
        return 0
    else:
        print("##########################################")
        print("### Start generating PyG graph of " + ("session2" if "session2" in args.storefile else "session1"))
        print("##########################################\n")
        graph,idx = read_node_atts(node_file, pyg_file, label_file)

    print("Start loading edge information")
    print("Start loading edge information",len(idx))

    process = tqdm(total=edge_size)
    edges = torch.zeros(2, edge_size,dtype=torch.int64)
    #edge_type = []
    count = 0
    with open(edge_file, 'r') as rf:
        while True:
            line = rf.readline()
            if line is None or len(line) == 0:
                break
            line_info = line.strip().split(",")
            source_id, dest_id,  source_type, dest_type, edge_type = line_info
            edges[0][count] = int(source_id)
            edges[1][count] = int(dest_id)
            #edge_type.append(edge_type)
            count += 1
            if count % 100000 == 0:
                process.update(100000)
    process.update(edge_size % 100000)
    process.close()
    print('Complete loading edge information\n')
    torch.save(edges,  "edges.pt")
    '''
    graph,idx = read_node_atts(node_file, pyg_file, label_file)
    edges = torch.load("edges.pt")

    print('Start converting edge information',idx)
    print(graph)

    graph.edge_index = edges
    print(graph)
    #idx,index_idx = idx.sort()

    #first layer

    nodes, adj_g = graph.sample_adj(idx, size=10)
    print(len(nodes))

    nodes, adj_g = graph.sample_adj(nodes, size=10)
    print(len(nodes))

    sub_graph = graph.subgraph(nodes,True)
    #sub_graph = graph.edge_subgraph(adj_g,True)

    print(sub_graph)

    print('Complete converting edge information\n')
    print('Start saving into pyg data')
    #torch.save(sub_graph, pyg_file + ".pt")
    torch.save(sub_graph,"subgraph.pt")

    print('Complete saving into pyg data\n')

    print("##########################################")
    print("### Complete generating PyG graph of " "session1")
    print("##########################################")
    return sub_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default="icdm2022_session1_train/icdm2022_session1_edges.csv")
    parser.add_argument('--node', type=str, default="icdm2022_session1_train/icdm2022_session1_nodes.csv")
    parser.add_argument('--label', type=str, default="icdm2022_session1_train/icdm2022_session1_train_labels.csv")
    parser.add_argument('--storefile', type=str, default="dataset")
    parser.add_argument('--reload', type=bool, default=False, help="Whether node features should be reloaded")
    args = parser.parse_args()
    if "session2" in args.storefile:
        edge_size = 120691444
        node_size = 10284026
    else:
        edge_size = 157814864
        node_size = 13806619
    if args.graph is not None and args.storefile is not None and args.node is not None:
        format_pyg_graph(args.graph, args.node, args.storefile, args.label)
        # read_node_atts(args.node, args.storefile, args.label)
