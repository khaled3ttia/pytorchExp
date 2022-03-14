import csv
import pandas as pd
import glob
import pathlib
import programl as pg
from programl.util.py import pbutil
from programl.proto import program_graph_pb2
from torch_geometric.data import Data
import torch

VOCABULARY = 'vocab/programl.csv'

'''
    loads programl vocabulary file into a dictionary

'''
def load_vocabulary(path):
    vocab = {}
    with open(path) as f:
        vocab_file = csv.reader(f.readlines(), delimiter='\t')
        for i, row in enumerate(vocab_file, start=-1):
            if i == -1:
                continue
            (_, _, _, text) = row
            vocab[text] = i
    return vocab

''' 
    This function is adopted from an older commit of PrograML 
    https://github.com/ChrisCummins/ProGraML/blob/2876ad5cf0cba2ab52c8b34fafa68a0af50c9d0d/programl/task/graph_level_classification/dataset.py

    It takes an input programl graph and vocabulary dictionary, 
    and produces pytorch data
'''
def convertGraph(graph, vocabulary):

    # edge index
    edge_tuples = [(edge.source, edge.target) for edge in graph.edge]
    edge_index = torch.tensor(edge_tuples).t().contiguous()
    
    # edge_attr
    positions = torch.tensor([edge.position for edge in graph.edge])
    flows = torch.tensor([int(edge.flow) for edge in graph.edge])

    edge_attr = torch.cat([flows, positions]).view(2, -1).t().contiguous()

    vocabulary_indices = vocab_ids = [
            vocabulary.get(node.text, len(vocabulary)) for node in graph.node
            ]
    
    xs = torch.tensor(vocabulary_indices)
    types = torch.tensor([int(node.type) for node in graph.node])
    
    x = torch.cat([xs, types]).view(2, -1).t().contiguous()
    data_dict = { "x": x, "edge_index": edge_index, "edge_attr": edge_attr}

    graph_features = graph.features.feature
    data_dict["y"] = torch.tensor(graph_features["devmap_label"].int64_list.value[0]).view(1)

    data_dict["aux_in"] = torch.tensor(
            [
                graph_features["transfer_bytes"].int64_list.value[0],
                graph_features["wgsize"].int64_list.value[0],
                graph_features["transfer_bytes_log1p"].float_list.value[0],
                graph_features["wgsize_log1p"].float_list.value[0]
            ]
        )

    data = Data(**data_dict)
    return data

def main():
    
    data_list = []
    
    vocab = load_vocabulary(pathlib.Path(VOCABULARY))
    for g in glob.glob('./graphs_*'):
        for f in glob.glob(f'{g}/*.pb'):
            #print(f'Reading graph {f}')

            gr = pbutil.FromFile(pathlib.Path(f'{f}'), program_graph_pb2.ProgramGraph())

            data_list.append(convertGraph(gr, vocab))

if __name__ == '__main__':
    main()
