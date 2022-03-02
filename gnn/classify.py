import torch
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset  # for ENZYMES dataset
from torch_geometric.datasets import Planetoid # for Cora dataset

edge_index = torch.tensor([[0,1],
                           [1,0],
                           [1,2],
                           [2,1]], dtype=torch.long)

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())

print(data)

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

print(len(dataset))

print(dataset.num_classes)

print(dataset.num_node_features)

dataset = dataset.shuffle()

train_dataset = dataset[:540]
print(train_dataset)
test_dataset = dataset[540:]
print(test_dataset)

# Try the Cora dataset 

dataset = Planetoid(root='/tmp/Cora', name='Cora')

print(dataset)

print(len(dataset))

print(dataset.num_classes)

print(dataset.num_node_features)

data = dataset[0]

print(data)

print(data.is_undirected)

print(data.train_mask.sum().item())

print(data.val_mask.sum().item())

print(data.test_mask.sum().item())

