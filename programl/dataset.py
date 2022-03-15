# better dataloader
import csv
import enum
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.data import Data, InMemoryDataset

from programl.proto.program_graph_pb2 import ProgramGraph

# make this file executable from anywhere

#full_path = os.path.realpath(__file__)
full_path = os.getcwd()
# print(full_path)
#REPO_ROOT = full_path.rsplit("ProGraML", maxsplit=1)[0] + "ProGraML"
# print(REPO_ROOT)
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, REPO_ROOT)
#REPO_ROOT = Path(REPO_ROOT)


# The vocabulary files used in the dataflow experiments.
#PROGRAML_VOCABULARY = REPO_ROOT / "deeplearning/ml4pl/poj104/programl_vocabulary.csv"
PROGRAML_VOCABULARY = full_path +  "/vocab/programl.csv"
#CDFG_VOCABULARY = REPO_ROOT / "deeplearning/ml4pl/poj104/cdfg_vocabulary.csv"
#assert PROGRAML_VOCABULARY.is_file(), f"File not found: {PROGRAML_VOCABULARY}"
#assert CDFG_VOCABULARY.is_file(), f"File not found: {CDFG_VOCABULARY}"

# The path of the graph2cdfg binary which converts ProGraML graphs to the CDFG
# representation.
#
# To build this file, clone the ProGraML repo and build
# //programl/cmd:graph2cdfg:
#
#   1.  git clone https://github.com/ChrisCummins/ProGraML.git
#   2.  cd ProGraML
#   3.  git checkout 2d93e5e14bf321336f1928d3364e9d7196cee995
#   4.  bazel build -c opt //programl/cmd:graph2cdfg
#   5.  cp -v bazel-bin/programl/cmd/graph2cdfg ${THIS_DIR}
#
#GRAPH2CDFG = REPO_ROOT / "deeplearning/ml4pl/poj104/graph2cdfg"
#assert GRAPH2CDFG.is_file(), f"File not found: {GRAPH2CDFG}"


def load(file: str, cdfg: bool = False) -> ProgramGraph:
    """Read a ProgramGraph protocol buffer from file.

    Args:
        file: The path of the ProgramGraph protocol buffer to load.
        cdfg: If true, convert the graph to CDFG during load.
    Returns:
        graph: the proto of the programl / CDFG graph
        orig_graph: the original programl proto (that contains graph level labels)
    """
    graph = ProgramGraph()
    with open(file, "rb") as f:
        proto = f.read()

    if cdfg:
        # hotfix missing graph labels in cdfg proto
        orig_graph = ProgramGraph()
        orig_graph.ParseFromString(proto)

        graph2cdfg = subprocess.Popen(
            [str(GRAPH2CDFG), "--stdin_fmt=pb", "--stdout_fmt=pb"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        proto, _ = graph2cdfg.communicate(proto)
        assert not graph2cdfg.returncode, f"CDFG conversion failed: {file}"

    graph.ParseFromString(proto)

    if not cdfg:
        orig_graph = graph
    return graph, orig_graph


def load_vocabulary(path: Path):
    """Read the vocabulary file used in the dataflow experiments."""
    vocab = {}
    with open(path) as f:
        vocab_file = csv.reader(f.readlines(), delimiter="\t")
        for i, row in enumerate(vocab_file, start=-1):
            if i == -1:  # Skip the header.
                continue
            (_, _, _, text) = row
            vocab[text] = i

    return vocab


class AblationVocab(enum.IntEnum):
    # No ablation - use the full vocabulary (default).
    NONE = 0
    # Ignore the vocabulary - every node has an x value of 0.
    NO_VOCAB = 1
    # Use a 3-element vocabulary based on the node type:
    #    0 - Instruction node
    #    1 - Variable node
    #    2 - Constant node
    NODE_TYPE_ONLY = 2


def getfilename(
    split: str, cdfg: bool = False, ablation_vocab: AblationVocab = AblationVocab.NONE
):
    """Generate the name for a data file.

    Args:
        split: The name of the split.
        cdfg: Whether using CDFG representation.
        ablate_vocab: The ablation vocab type.

    Returns:
        A file name which uniquely identifies this combination of
        split/cdfg/ablation.
    """
    name = str(split)
    if cdfg:
        name = f"{name}_cdfg"
    if ablation_vocab != AblationVocab.NONE:
        # transform if ablation_vocab was passed as int.
        if type(ablation_vocab) == int:
            ablation_vocab = AblationVocab(ablation_vocab)
        name = f"{name}_{ablation_vocab.name.lower()}"
    return f"{name}_data.pt"


def nx2data(
    graph: ProgramGraph,
    vocabulary: Dict[str, int],
    y_feature_name: Optional[str] = None,
    ignore_profile_info=True,
    ablate_vocab=AblationVocab.NONE,
    orig_graph: ProgramGraph = None,
):
    r"""Converts a program graph protocol buffer to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        graph           A program graph protocol buffer.
        vocabulary      A map from node text to vocabulary indices.
        y_feature_name  The name of the graph-level feature to use as class label.
        ablate_vocab    Whether to use an ablation vocabulary.
        orig_graph      A program graph protocol buffer that has graph level labels.
    """

    # collect edge_index
    edge_tuples = [(edge.source, edge.target) for edge in graph.edge]
    edge_index = torch.tensor(edge_tuples).t().contiguous()

    # collect edge_attr
    positions = torch.tensor([edge.position for edge in graph.edge])
    flows = torch.tensor([int(edge.flow) for edge in graph.edge])

    edge_attr = torch.cat([flows, positions]).view(2, -1).t().contiguous()

    # collect x
    if ablate_vocab == AblationVocab.NONE:
        vocabulary_indices = vocab_ids = [
            vocabulary.get(node.text, len(vocabulary)) for node in graph.node
        ]
    elif ablate_vocab == AblationVocab.NO_VOCAB:
        vocabulary_indices = [0] * len(graph.node)
    elif ablate_vocab == AblationVocab.NODE_TYPE_ONLY:
        vocabulary_indices = [int(node.type) for node in graph.node]
    else:
        raise NotImplementedError("unreachable")

    xs = torch.tensor(vocabulary_indices)
    types = torch.tensor([int(node.type) for node in graph.node])

    x = torch.cat([xs, types]).view(2, -1).t().contiguous()

    assert (
        edge_attr.size()[0] == edge_index.size()[1]
    ), f"edge_attr={edge_attr.size()} size mismatch with edge_index={edge_index.size()}"

    data_dict = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }

    # maybe collect these data too
    if y_feature_name is not None:
        assert orig_graph is not None, "need orig_graph to retrieve graph level labels!"
        y = torch.tensor(
            orig_graph.features.feature[y_feature_name].int64_list.value[0]
        ).view(
            1
        )  # <1>
        if y_feature_name == "poj104_label":
            y -= 1
        data_dict["y"] = y

    # branch prediction / profile info specific
    if not ignore_profile_info:
        raise NotImplementedError(
            "profile info is not supported with the new nx2data (from programgraph) adaptation."
        )
        profile_info = []
        for i, node_data in nx_graph.nodes(data=True):
            # default to -1, -1, -1 if not all profile info is given.
            if not (
                node_data.get("llvm_profile_true_weight") is not None
                and node_data.get("llvm_profile_false_weight") is not None
                and node_data.get("llvm_profile_total_weight") is not None
            ):
                mask = 0
                true_weight = -1
                false_weight = -1
                total_weight = -1
            else:
                mask = 1
                true_weight = node_data["llvm_profile_true_weight"]
                false_weight = node_data["llvm_profile_false_weight"]
                total_weight = node_data["llvm_profile_total_weight"]

            profile_info.append([mask, true_weight, false_weight, total_weight])

        data_dict["profile_info"] = torch.tensor(profile_info)

    # make Data
    data = Data(**data_dict)

    return data



class DevmapDataset(InMemoryDataset):
    def __init__(
        self,
        root="deeplearning/ml4pl/poj104/devmap_data",
        split="fail",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
        cdfg: bool = False,
        ablation_vocab: AblationVocab = AblationVocab.NONE,
    ):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.
            split: 'amd' or 'nvidia'
            cdfg: Use CDFG graph representation.
        """
        assert split in [
            "amd",
            "nvidia",
        ], f"Split is {split}, but has to be 'amd' or 'nvidia'"
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        self.cdfg = cdfg
        self.ablation_vocab = ablation_vocab
        self.transform = transform.Compose([transforms.ToTensor()])
        super().__init__(root, self.transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "devmap_data.zip"

    @property
    def processed_file_names(self):
        base = getfilename(self.split, self.cdfg, self.ablation_vocab)

        if tuple(self.train_subset) == (0, 100):
            return [base]
        else:
            return [
                f"{name}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def download(self):
        # download to self.raw_dir
        pass

    def return_cross_validation_splits(self, split):
        assert self.train_subset == [
            0,
            100,
        ], "Do cross-validation on the whole dataset!"
        # num_samples = len(self)
        # perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # 10-fold cross-validation
        n_splits = 10
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        (train_index, test_index) = list(kf.split(self.data.y, self.data.y))[split]


        train_data = torch.utils.data.Subset(self, train_index)
        test_data = torch.utils.data.Subset(self, test_index)
        return train_data, test_data

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(100, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def process(self):
        # check if we need to create the full dataset:
        name = getfilename(self.split, self.cdfg, self.ablation_vocab)
        full_dataset = Path(self.processed_dir) / name
        if full_dataset.is_file():
            print(
                f"Full dataset {full_dataset.name} found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        vocab = load_vocabulary(CDFG_VOCABULARY if self.cdfg else PROGRAML_VOCABULARY)
        assert len(vocab) > 0, "vocab is empty :|"

        root = Path(self.root)

        # Get list of source file names and attributes
        input_files = list((root / f"graphs_{self.split}").iterdir())

        num_files = len(input_files)
        print("\n--- Preparing to read", num_files, "input files")

        # read data into huge `Data` list.

        data_list = []
        for i in tqdm.tqdm(range(num_files)):
            filename = input_files[i]

            proto, _ = load(filename, cdfg=self.cdfg)
            data = nx2data(proto, vocabulary=vocab, ablate_vocab=self.ablation_vocab)

            # graph2cdfg conversion drops the graph features, so we may have to
            # reload the graph.
            if self.cdfg:
                proto = load(filename)

            # Add the features and label.
            proto_features = proto.features.feature
            data["y"] = torch.tensor(
                proto_features["devmap_label"].int64_list.value[0]
            ).view(1)
            data["aux_in"] = torch.tensor(
                [
                    proto_features["transfer_bytes"].int64_list.value[0],
                    proto_features["wgsize"].int64_list.value[0],
                ]
            )

            data_list.append(data)

        ##################################

        print(f" * COMPLETED * === DATASET Devmap-{name}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET Devmap-{name}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100):
            self._save_train_subset()


class POJ104Dataset(InMemoryDataset):
    def __init__(
        self,
        root="deeplearning/ml4pl/poj104/classifyapp_data",
        split="fail",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
        cdfg: bool = False,
        ablation_vocab: AblationVocab = AblationVocab.NONE,
    ):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.
            cdfg: Use the CDFG graph format and vocabulary.
        """
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        self.cdfg = cdfg
        self.ablation_vocab = ablation_vocab
        super().__init__(root, transform, pre_transform)

        assert split in ["train", "val", "test"]
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "classifyapp_data.zip"  # ['ir_val', 'ir_val_programl']

    @property
    def processed_file_names(self):
        base = getfilename(self.split, self.cdfg, self.ablation_vocab)

        if tuple(self.train_subset) == (0, 100) or self.split in ["val", "test"]:
            return [base]
        else:
            assert self.split == "train"
            return [
                f"{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def download(self):
        # download to self.raw_dir
        pass

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        import numpy as np

        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(100, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def process(self):
        # hardcoded
        num_classes = 104

        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / filename(
            self.split, self.cdfg, self.ablation_vocab
        )
        if full_dataset.is_file():
            assert self.split == "train", "here shouldnt be reachable."
            print(
                f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        # get vocab first
        vocab = load_vocabulary(CDFG_VOCABULARY if self.cdfg else PROGRAML_VOCABULARY)
        assert len(vocab) > 0, "vocab is empty :|"
        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f"Creating {self.split} dataset at {str(ds_base)}")

        split_folder = ds_base / (self.split)
        assert split_folder.exists(), f"{split_folder} doesn't exist!"

        # collect .pb and call nx2data on the fly!
        print(
            f"=== DATASET {split_folder}: Collecting ProgramGraph.pb files into dataset"
        )

        # only take files from subfolders (with class names!) recursively
        files = [x for x in split_folder.rglob("*ProgramGraph.pb")]
        assert len(files) > 0, "no files collected. error."
        for file in tqdm.tqdm(files):
            # skip classes that are larger than what config says to enable debugging with less data
            # class_label = int(file.parent.name) - 1  # let classes start from 0.
            # if class_label >= num_classes:
            #    continue

            g, orig_graph = load(file, cdfg=self.cdfg)
            data = nx2data(
                graph=g,
                vocabulary=vocab,
                ablate_vocab=self.ablation_vocab,
                y_feature_name="poj104_label",
                orig_graph=orig_graph,
            )
            data_list.append(data)

        print(f" * COMPLETED * === DATASET {split_folder}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET {split_folder}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in [
            "val",
            "test",
        ]:
            self._save_train_subset()


class LegacyPOJ104Dataset(InMemoryDataset):
    def __init__(
        self,
        root="deeplearning/ml4pl/poj104/classifyapp_data",
        split="fail",
        transform=None,
        pre_transform=None,
        train_subset=[0, 100],
        train_subset_seed=0,
    ):
        """
        Args:
            train_subset: [start_percentile, stop_percentile)    default [0,100).
                            sample a random (but fixed) train set of data in slice by percent, with given seed.
            train_subset_seed: seed for the train_subset fixed random permutation.

        """
        self.split = split
        self.train_subset = train_subset
        self.train_subset_seed = train_subset_seed
        super().__init__(root, transform, pre_transform)

        assert split in ["train", "val", "test"]
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "classifyapp_data.zip"  # ['ir_val', 'ir_val_programl']

    @property
    def processed_file_names(self):
        base = f"{self.split}_data.pt"

        if tuple(self.train_subset) == (0, 100) or self.split in ["val", "test"]:
            return [base]
        else:
            assert self.split == "train"
            return [
                f"{self.split}_data_subset_{self.train_subset[0]}_{self.train_subset[1]}_seed_{self.train_subset_seed}.pt"
            ]

    def download(self):
        # download to self.raw_dir
        pass

    def _save_train_subset(self):
        """saves a train_subset of self to file.
        Percentile slice is taken according to self.train_subset
        with a fixed random permutation with self.train_subset_seed.
        """
        import numpy as np

        perm = np.random.RandomState(self.train_subset_seed).permutation(len(self))

        # take slice of perm according to self.train_subset
        start = np.math.floor(len(self) / 100 * self.train_subset[0])
        stop = np.math.floor(len(self) / 100 * self.train_subset[1])
        perm = perm[start:stop]
        print(f"Fixed permutation starts with: {perm[:min(100, len(perm))]}")

        dataset = self.__indexing__(perm)

        data, slices = dataset.data, dataset.slices
        torch.save((data, slices), self.processed_paths[0])
        return

    def process(self):
        # hardcoded
        num_classes = 104

        # check if we need to create the full dataset:
        full_dataset = Path(self.processed_dir) / f"{self.split}_data.pt"
        if full_dataset.is_file():
            assert self.split == "train", "here shouldnt be reachable."
            print(
                f"Full dataset found. Generating train_subset={self.train_subset} with seed={self.train_subset_seed}"
            )
            # just get the split and save it
            self.data, self.slices = torch.load(full_dataset)
            self._save_train_subset()
            print(
                f"Saved train_subset={self.train_subset} with seed={self.train_subset_seed} to disk."
            )
            return

        # ~~~~~ we need to create the full dataset ~~~~~~~~~~~
        assert not full_dataset.is_file(), "shouldnt be"
        processed_path = str(full_dataset)

        # read data into huge `Data` list.
        data_list = []

        ds_base = Path(self.root)
        print(f"Creating {self.split} dataset at {str(ds_base)}")
        # TODO change this line to go to the new format
        out_base = ds_base / ("ir_" + self.split + "_programl")
        assert out_base.exists(), f"{out_base} doesn't exist!"
        # TODO collect .ll.pickle instead and call nx2data on the fly!
        print(f"=== DATASET {out_base}: Collecting .data.p files into dataset")

        folders = [
            x
            for x in out_base.glob("*")
            if x.is_dir() and x.name not in ["_nx", "_tuples"]
        ]
        for folder in tqdm.tqdm(folders):
            # skip classes that are larger than what config says to enable debugging with less data
            if int(folder.name) > num_classes:
                continue
            for k, file in enumerate(folder.glob("*.data.p")):
                with open(file, "rb") as f:
                    data = pickle.load(f)
                data_list.append(data)

        print(f" * COMPLETED * === DATASET {out_base}: now pre-filtering...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        print(
            f" * COMPLETED * === DATASET {out_base}: Completed filtering, now pre_transforming..."
        )

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), processed_path)

        # maybe save train_subset as well
        if not tuple(self.train_subset) == (0, 100) and self.split not in [
            "val",
            "test",
        ]:
            self._save_train_subset()


if __name__ == "__main__":
    # d = NewNCCDataset()
    # print(d.data)
    root = "/home/zacharias/llvm_datasets/threadcoarsening_data/"
    #a = ThreadcoarseningDataset(root, "Cypress")
    #b = ThreadcoarseningDataset(root, "Tahiti")
    #c = ThreadcoarseningDataset(root, "Fermi")
    #d = ThreadcoarseningDataset(root, "Kepler")
