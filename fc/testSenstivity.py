import sys
import argparse
import torch
import torch.nn as nn
from classify import Net

FACTOR = 8
NO_CLASSES = 4
N_FEATURES = 4

def prepareInput(inputStr, nfeatures):
    # convert to int
    intInput = [int(ch) for ch in inputStr]

    # convert to torch 
    intInput = torch.tensor(intInput)

    # reshape
    intInput = torch.reshape(intInput, (1, nfeatures))

    return intInput.float()

def test8():
    sampleInputs = ['12344320', '12211332', '11203221', '56577879', '32323132', '94565694']
    expected =  [0, 2, -1, 3, 3, 0]

    for i in range(len(sampleInputs)):
        prepInput = prepareInput(sampleInputs[i], 8)
        output = net(prepInput)
        predicted = int(torch.argmax(output))
        print(f'{sampleInputs[i]}:{expected[i]}:{predicted}')

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()

    parser.add_argument("--nnFactor", type=int, help="Factor used during NN construction for hidden layers")
    parser.add_argument("--n_classes", type=int, help="Number of classes (labels)")

    parser.add_argument("--features", type=int, help="Number of features")


    args = parser.parse_args()

    if (args.features):
        N_FEATURES = args.features


    if (args.nnFactor):
        FACTOR = args.nnFactor

    if (args.n_classes):
        NO_CLASSES = args.n_classes

    modelFile = 'model-' + str(N_FEATURES) + '.pt'

    # Load the trained model
    net = Net(N_FEATURES, FACTOR , NO_CLASSES )

    net.load_state_dict(torch.load(modelFile))

    net.eval()

    print(f'Input String:Expected:Predicted')

    if (N_FEATURES == 8):
        test8()
    
    

