import argparse
import math
import pandas as pd
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

# Reset weights before each fold
# used in k-fold cross validation
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


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

    print(f'Training with learning rate of {lr} and {epochs} epochs...')
     
    # Create a KFold object (sklearn) with k=10
    kfold = KFold(n_splits=10, shuffle=True)

   
    accuracies = []

    # use k-fold cross validation to evaluate the accuracy is the model 
    # Note that k-fold cross validation is only used to evaluate the model
    # more accurately and remove some of the bias as compared to train-test
    # split only. However, all models obtained during cross validation are 
    # discarded. If the accuracy numbers obtained by cross validation are acceptable,
    # a new model is trained using the entire dataset with the same parameters
    # that turned out to be good.
    
    # K-fold cross validation is usually used in the experiments phase to choose between
    # different model and different parameters

    # for more info about this:
    #   [1] https://machinelearningmastery.com/k-fold-cross-validation/
    #   [2] https://www.researchgate.net/post/How_to_select_the_classification_model_after_k-cross-validation (specifically Roberto Vega answer to the question)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'Fold: {fold}')
        
        # randomly sample elements from ids 
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
        # data loaders based on samples
        trainset = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=train_subsampler)
        testset = torch.utils.data.DataLoader(dataset, batch_size=10, sampler=test_subsampler)


        # instantiate the neural net
        net = Net(NO_FEATURES, FACTOR, NO_CLASSES)
        
        # reset weights
        net.apply(reset_weights)

        # define the loss function and the optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(trainset):
                output = net(inputs.float())
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        fold_accuracy = check_accuracy(testset, net)
        print(f'Accuracy for fold {fold} : {fold_accuracy}')
        accuracies.append(fold_accuracy)

    avg_accuracy = sum(accuracies)/len(accuracies)
    print(f'Average accuracy accross all folds: {avg_accuracy}')

    # train the final model with the entire dataset
    # this article [https://machinelearningmastery.com/train-final-machine-learning-model/] 
    # is a nice guide to this procedure 
    
    # use the entire dataset
    final_trainset = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

    # create the model using the previous (good) parameters
    net = Net(NO_FEATURES, FACTOR, NO_CLASSES)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    print(f'Training the final model on the entire dataset...')
    for epoch in range(epochs):
        for i, (input, labels) in enumerate(final_trainset):
            output = net(inputs.float())
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # save the model
    torch.save(net.state_dict(), 'model-' +str(NO_FEATURES) + '.pt')
    print(f'Model saved successfull to model-{str(NO_FEATURES)}.pt!')
