import time
import numpy as np
import csv
import pandas as pd
import glob
import pathlib
import programl as pg
from programl.util.py import pbutil
from programl.proto import program_graph_pb2
from torch_geometric.data import Data
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import modeling
from modeling import GGNNModel
from dataset import DevmapDataset
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

class GGNN_Devmap_Config(object):
    def __init__(self):
        self.name = self.__class__.__name__
        self.max_num_nodes = 200000
        self.patience = 10000
        self.clip_grad_norm = 0.0
        self.train_subset = [0, 100]
        self.random_seed = 42

        self.output_dropout = 0.0

        self.emb_size = 200
        self.edge_type_count = 3

        self.vocab_size = 8568
        self.ablation_vocab = 0 


        # Model Hyperparametrs
        self.gnn_layers = 8
        self.message_weight_sharing = 2
        self.update_weight_sharing = 2

        # node types 0 and 1 for stmts and ids
        self.use_nod_types = True
        self.use_edge_bias = True
        self.position_embeddings = True

        # Aggregate by mean or by sum
        self.msg_mean_aggregation = True
        self.backward_edges = True

        # Regularization
        self.edge_weight_dropout = 0.0
        self.graph_state_dropout = 0.2
        

        self.batch_size = 64
        self.lr = 2.5e-4
        self.num_epochs = 150 
        self.graph_state_dropout = 0.0

        self.aux_use_better = False
        self.intermediate_loss_weight = 0.2
        self.aux_in_size = 2
        self.aux_in_layer_size = 32 
        self.aux_in_log1p = True

        self.num_classes = 2
        self.has_graph_labels = True
        self.has_aux_input = True

        self.inst2vec_embeddings = "random"

        # ?? 
        self.selector_size = 0
        self.hidden_size = self.emb_size + self.selector_size

'''
def return_cross_validation_splits(data, current_kfold_split):
    n_splits = 10
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_seed=42)
    (train_index, test_index) = list(kf.split(
'''

class Learner(object):
    def __init__(self, dataset, current_kfold_split):
        
        Model, Config = GGNNModel, GGNN_Devmap_Config
        #self.train_data = data[0]
        #self.valid_data = data[1]
        self.global_training_step = 0 
        self.current_epoch = 1

        # get config 
        self.config = Config() 

        # TODO: Model ? 
        self.model = Model(self.config, False) 

        
        self.load_data(dataset, 10, current_kfold_split)

    def data2input(self, batch):

        num_graphs = bath.batch[-1].item() + 1

        edge_lists = []
        edge_positions = []

        edge_indices = list(range(3))

        for i in edge_indices:
            mask = batch.edge_attr[:, 0] == i
            edge_list = batch.edge_index[:, mask].t()
            edge_lists.append(edge_list)

            edge_pos = batch.edge_attr[mask, 1]
            edge_positions.append(edge_pos)


        inputs = {
                "vocab_ids": batch.x[:, 0],
                "edge_lists": edge_lists, 
                "pos_lists": edge_positions, 
                "num_graphs": num_graphs,
                "graph_nodes_list": batch.batch,
                "node_types": batch.x[:,1]
                }
        if batch.y is not None:
            inputs.update(
                    {
                        "labels": batch.y
                    }
                    )

        inputs.update({"aux_in": batch.aux_in.to(dtype=torch.float)})
        inputs.update({"runtimes": batch.runtimes.to(dtype=torch.fload)})

        return inputs
    
    def load_data(self, dataset, kfold, current_kfold_split):
        """Set self.train_data, self.test_data, self.valid_data depending on the dataset used."""
        if not kfold:
            assert current_kfold_split is None
        if "_" in dataset:
            split = dataset.rsplit("_", maxsplit=1)[-1]
        Dataset, data_dir = DevmapDataset, "devmap_data"
        self.data_dir = data_dir

        # Switch cases by dataset
        # ~~~~~~~~~~ NCC ~~~~~~~~~~~~~~~~~~~~~
       # ~~~~~~~~~~ POJ 104 ~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~ DEVMAP ~~~~~~~~~~~~~~~~~~~~~
        if dataset in ["devmap_amd", "devmap_nvidia"]:
            assert (
                kfold and current_kfold_split is not None
            ), "Devmap only supported with kfold flag!"
            assert current_kfold_split < 10
            # get the whole dataset then get the correct split
            ds = Dataset(
                root=self.data_dir,
                split=split,
                train_subset=self.config.train_subset,
                cdfg=False,
                ablation_vocab=self.config.ablation_vocab,
            )
            train_dataset, valid_dataset = ds.return_cross_validation_splits(
                current_kfold_split
            )

            self.train_data = None
            self.valid_data = torch.utils.data.DataLoader(
                valid_dataset, batch_size=self.config.batch_size * 2, shuffle=False
            )

            # only maybe set train_data.
            
            self.train_data = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    )
            self.test_data = None


    def run_epoch(self, loader, epoch_type):
        epoch_loss, epoch_accuracy = 0, 0
        epoch_actual_rt, epoch_optimal_rt = 0,0
        start_time = time.time()
        processed_graphs = 0 
        predicted_targets = 0 

        for step, batch in enumerate(loader):
            inputs = self.data2input(batch)
            num_graphs = inputs["num_graphs"]
            num_targets = num_graphs

            predicted_targets += num_targets
            processed__graphs += num_graphs

            if (epoch_type == 'train'):
                self.global_training_step += 1
                # TODO: model
                if not self.model.training:
                    self.model.train()
                outputs = self.model(**inputs)
            else:
                if self.model.training:
                    self.model.eval()
                    self.model.opt.zero_grad()
                with torch.no_grad():
                    outputs = self.model(**inputs)

            if hasattr(batch, "runtimes"):

                (
                    logits,
                    accuracy,
                    correct,
                    targets,
                    actual_rt,
                    optimal_rt,
                    graph_features,
                    *unroll_stats
                ) = outputs
                epoch_actual_rt += torch.sum(actual_rt).item()
                epoch_optimal_rt += torch.sum(optimal_rt).item()
            else:
                (
                    logits,
                    accuracy,
                    correct,
                    targets,
                    graph_features,
                    *unroll_stats
                ) = outputs
            loss = self.model.loss((logits, graph_features), targets)

            epoch_loss += loss.item() * num_targets
            epoch_accuracy += accuracy.item() * num_targets

            if epoch_type == 'train':
                loss.backward()
                if self.model.config.clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm(
                            self.model.parameters(), self.model.config.clip_grad_norm
                    )
                self.model.opt.step()
                self.model.opt.zero_grad()

        mean_loss = epoch_loss / predicted_targets
        mean_accuracy = epoch_accuracy / predicted_targets
        instance_per_sec = processed_graphs / (time.time() - start_time)
        epoch_preplexity = np.exp(mean_loss)

        returns = (
            mean_loss,
            mean_accuracy,
            instance_per_sec,
            epoch, perplexity,
            epoch_actual_rt,
            epoch_optimal_rt
        )

        return returns

    

    def train(self):
        log_to_save = [] 
        total_time_start = time.time()

        (best_val_acc, best_val_epoch) = (0.0, 0)
        target_epoch = self.current_epoch + self.config.num_epochs
        for epoch in range(self.current_epoch, target_epoch):
            print(f'== Epoch {epoch}/{target_epoch}')
            (
                train_loss,
                train_acc,
                train_speed,
                train_ppl,
                train_art,
                train_ort
            ) = self.run_epoch(self.train_data, "train")

            print('Train: loss: {train_loss} | acc: {train_acc} | ppl: {train_ppl} | instances/sec: {train_speed} | runtime: {train_art} | opt: {train_ort}')

            (
                valid_loss,
                valid_acc,
                valid_speed,
                valid_ppl,
                valid_art,
                valid_org
            ) = self.run_epoch(self.valid_data, "eval")
            print('Valid: loss: {valid_loss} | acc: {valid_acc} | ppl: {valid_ppl} | instances/sec: {valid_speed} | runtime: {valid_art} | opt: {valid_ort}')

            epoch_time = time.time() - total_time_start
            self.current_epoch = epoch

        #self.save_model(epoch)



def main():
    
    test_mode = False

    data_list = []
    
    vocab = load_vocabulary(pathlib.Path(VOCABULARY))

    '''
    for g in glob.glob('./graphs_*'):
        for f in glob.glob(f'{g}/*.pb'):
            #print(f'Reading graph {f}')

            gr = pbutil.FromFile(pathlib.Path(f'{f}'), program_graph_pb2.ProgramGraph())

            data_list.append(convertGraph(gr, vocab))
    '''
    num_splits = 10
    dataset = "devmap_amd"
    config = GGNN_Devmap_Config()

    for split in range(num_splits):
        learner = Learner(
                dataset=dataset,
                current_kfold_split=split
            )
        if len(learner.valid_data) == 0:
            print('skipping validation')
        learner.test() if test_mode else learner.train()
    

    '''
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
    for split, (train_index, test_index) in enumerate(kf.split(data_list)):
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_index)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_index)
        
        # Potential error : removed batch_size from here
        #train_dataset = torch.utils.data.DataLoader(data_list, sampler=train_subsampler)
        #valid_dataset = torch.utils.data.DataLoader(data_list, sampler=test_subsampler)
        
        train_dataset = data_list[train_index,:]
        
        train_data = None
        valid_data = torch.utils.data.DataLoader(valid_dataset.dataset, batch_size=config.batch_size*2, shuffle=False)

        if (not test_mode):
            train_data = torch.utils.data.DataLoader(train_dataset.dataset, batch_size=config.batch_size, shuffle=True)


        learner = Learner([train_data, valid_data])

        learner.test() if test_mode else learner.train()

    '''
if __name__ == '__main__':
    main()
