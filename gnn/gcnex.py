import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid 


class GCN(torch.nn.Module):
    def __init__(self, no_features, no_classes):
        super().__init__()

        self.conv1 = GCNConv(no_features, 16)
        self.conv2 = GCNConv(16, no_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':

    dataset = Planetoid(root='/tmp/Cora', name = 'Cora')
    
    NUM_FEATURES = dataset.num_node_features
    NUM_CLASSES = dataset.num_classes

    model = GCN(NUM_FEATURES, NUM_CLASSES)

    data = dataset[0]
    print(f'Size of graph dataset is {len(dataset)} graphs')


    # node feature matrix
    print('Node feature matrix')
    print(data.x)
    print(data.x.shape)

    # edge index (graph connectivity)
    print('Edge index')
    print(data.edge_index)


    # edge feature matrix
    print(data.edge_attr)

    # labels
    print(data.y)
    # print unique values
    print(torch.unique(data.y))

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay=5e-4)
    model.train()

    epochs = 200

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())

    print(f'Accuracy: {acc: .4f}')

