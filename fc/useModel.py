import sys
import argparse
import torch
import torch.nn as nn
from classify import Net

#INPUT = "121121553553"
#INPUT = "123456654321"
#INPUT = "123456765432"
#INPUT = "123320123320"
#INPUT = "12345543201234554320"
INPUT = "12345543211234554320"
FACTOR = 8
NO_CLASSES = 4

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, help="Input string to predict the class for")
    parser.add_argument("--nnFactor", type=int, help="Factor used during NN construction for hidden layers")
    parser.add_argument("--n_classes", type=int, help="Number of classes (labels)")

    args = parser.parse_args()

    if (args.input):
        INPUT = args.input 

    if (args.nnFactor):
        FACTOR = args.nnFactor

    if (args.n_classes):
        NO_CLASSES = args.n_classes

    # Load the trained model
    net = Net(len(INPUT), FACTOR , NO_CLASSES )

    net.load_state_dict(torch.load('model.pt'))

    net.eval()

    # Find out the number of features
    NUM_FEATURES = len(INPUT)

    # generate the compatible tensor for input

    intInput = [int(ch) for ch in INPUT]

    inputVect = torch.tensor(intInput)

    inputVect = torch.reshape(inputVect, (1,NUM_FEATURES))

    # Find the predicted output 
    output = net(inputVect.float())
    
    class_no = torch.argmax(output)
        
    print(f'{INPUT}:{int(class_no)}')
