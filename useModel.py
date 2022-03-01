import sys
import torch
import torch.nn as nn
from classify import Net

#INPUT = "121121553553"
#INPUT = "123456654321"
#INPUT = "123456765432"
#INPUT = "123320123320"
#INPUT = "12345543201234554320"
INPUT = "12345543211234554320"

if __name__ == '__main__':
    
    # get command line input if any 
    no_args = len(sys.argv)
    if (no_args > 1):
        INPUT = sys.argv[1]

    # Load the trained model
    net = Net()

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
        
    print(int(class_no))
