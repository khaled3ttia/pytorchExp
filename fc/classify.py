import argparse
import math
import pandas as pd
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# default values, can be overriden later
#NO_FEATURES = 10
#NO_CLASSES = 4
#FACTOR = 8

# Prepare the dataset
class simpleDataset(Dataset):
    def __init__(self, datafile):
        data = pd.read_csv(datafile, header=None)
        self.x = torch.tensor(data.iloc[:,:-1].values)
        self.y = torch.tensor(data.iloc[:,-1].values)

        self.n_samples = data.shape[0]
        self.n_features = data.shape[1] - 1
        self.n_classes = data.iloc[:,-1].nunique()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

# Create the model
class Net(nn.Module):
    def __init__(self, NO_FEATURES, FACTOR, NO_CLASSES):
        super(Net, self).__init__()
        # define the fully-connected layers

        # input layer
        self.fc1 = nn.Linear(NO_FEATURES, math.ceil(NO_FEATURES*FACTOR))

        # hidden layers
        self.fc2 = nn.Linear(math.ceil(NO_FEATURES*FACTOR), math.ceil(NO_FEATURES*FACTOR))
        self.fc3 = nn.Linear(math.ceil(NO_FEATURES*FACTOR), math.ceil(NO_FEATURES*FACTOR))

        # output layer
        self.fc4 = nn.Linear(math.ceil(NO_FEATURES*FACTOR), NO_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)

# loader is either train or testset
# model is the NN
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0 

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            scores = model(x.float())
            for idx, i in enumerate(scores):
                if torch.argmax(i) == y[idx]:
                    num_correct += 1
                num_samples += 1
    return (num_correct * 1.0/num_samples)

if __name__ == '__main__':
    
    # default input dataset file (if not specified by user)

    datafile = 'data.csv'
    

    # parse command line arguments

    parser = argparse.ArgumentParser()

    parser.add_argument("--datafile", type=str, help="Path of dataset file")
    parser.add_argument("--train_size", type=float, help="Size for the training set as a percent of total dataset size")
    parser.add_argument("--epochs", type=int, help="Number of epochs for the training phase")
    parser.add_argument("--lrate", type=float, help="Learning Rate")
    parser.add_argument("--nnFactor", type=int, help="The factor applied to the number of features to specify the number of nodes in the NN hidden layers")

    args = parser.parse_args()

    if (args.datafile):
        datafile = args.datafile

    print(f'Loading dataset file {datafile}...')
    # initialize the dataset object
    dataset = simpleDataset(datafile)

    # extract number of features and classes from the dataset 
    NO_FEATURES = dataset.n_features
    NO_CLASSES = dataset.n_classes
    print(f'Successfully loaded! Number of features is {NO_FEATURES}\nand number of labels is {NO_CLASSES}')

    # set sizes for train and test 
    train_size = int(0.8 * len(dataset))

    if (args.train_size):
        train_size = int(args.train_size * len(dataset))

    test_size = len(dataset) - train_size
    print(f'Splitting data set as follows: {(train_size/len(dataset))*100}% Training, {(test_size/len(dataset))*100}% Test...')

    # TODO: add cross-validation

    # actually randomly split train and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    # create a data loader for each of the train and test sets
    trainset = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

    testset = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

    # learning rate 
    lr = 0.01
    if (args.lrate):
        lr = args.lrate

    # number of training ephocs
    epochs = 40
    if (args.epochs):
        epochs = args.epochs

    FACTOR = 8
    if (args.nnFactor):
        FACTOR = args.nnFactor

    # instantiate the neural net
    net = Net(NO_FEATURES, FACTOR, NO_CLASSES)

    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    print(f'Training with learning rate of {lr} and {epochs} epochs...')

    # do the training
    for epoch in range(epochs):

        for i, (inputs, labels) in enumerate(trainset):
            output = net(inputs.float())
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # find out the accuracy for both the training and testing sets
    train_acc = check_accuracy(trainset, net)
    test_acc = check_accuracy(testset, net)

    print(f'Training set accuracy: {train_acc}\nTest set accuracy: {test_acc}')

    # save the model
    torch.save(net.state_dict(), 'model.pt')


