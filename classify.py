import math
import pandas as pd
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

NO_FEATURES = 12
NO_CLASSES = 4


class simpleDataset(Dataset):
    def __init__(self):
        data = pd.read_csv('data.csv', header=None)
        self.x = torch.tensor(data.iloc[:,:-1].values)
        self.y = torch.tensor(data.iloc[:,-1].values)

        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

# Create the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # define the fully-connected layers

        # input layer
        self.fc1 = nn.Linear(NO_FEATURES, math.ceil(NO_FEATURES*4))

        # hidden layers
        self.fc2 = nn.Linear(math.ceil(NO_FEATURES*4), math.ceil(NO_FEATURES*4))
        self.fc3 = nn.Linear(math.ceil(NO_FEATURES*4), math.ceil(NO_FEATURES*4))

        # output layer
        self.fc4 = nn.Linear(math.ceil(NO_FEATURES*4), NO_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x)





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
    return (num_correct/num_samples)

if __name__ == '__main__':
    
    # initialize the dataset object
    dataset = simpleDataset()

    # set sizes for train and test 
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # actually randomly split train and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    # create a data loader for each of the train and test sets
    trainset = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

    testset = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)

    # learning rate 
    lr = 0.01

    # number of training ephocs
    epochs = 20

    # instantiate the neural net
    net = Net()

    # define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

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


