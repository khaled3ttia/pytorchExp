import torch
import torch.nn as nn
from classify import Net

if __name__ == '__main__':

    net = Net()

    net.load_state_dict(torch.load('model.pt'))

    net.eval()

    inputVect = torch.tensor([1, 2, 1, 1, 2, 1, 5, 5, 3, 5, 5, 3])

    output = net(inputVect.float())

    class_no = torch.argmax(output)
        

    print(class_no)
