##
##  Author: Khaled Abdelaal
##          khaled.abdelaal@ou.edu
##
##          A simple script to generate a dataset of NUM_FEATURES features
##              and NUM_RECORDS records with some simple patterns in each
##              of the generated records
##
###############################################################
import argparse
import random

# default values (if not specified by command line options)
NUM_FEATURES = 12 # For now, it must be a multiple of 4
NUM_RECORDS = 1200
DATASET_FILE='data' # dataset filename

# A function to add an item to the set
# returns true if the item was actually added,
# otherwise false
def setAdd(s, item):
    l1 = len(s)
    s.add(item)
    return len(s) > l1

if __name__ == '__main__':

    # Parse commandline arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--features", type=int, help='Number of features')
    parser.add_argument("--records", type=int, help='Number of records')
    parser.add_argument("--file", type=str, help='Output filename')

    args = parser.parse_args()

    if (args.features):
        if (args.features % 4 != 0):
            print("Invalid number of features, using a default of 12")
        else:
            NUM_FEATURES = args.features

    if (args.records):
        NUM_RECORDS = args.records

    if (args.file):
        DATASET_FILE = args.file
    

    # Stores input instances
    records = set()

    # generate random input
    for i in range(NUM_RECORDS):
        #records.append([])
        tmp = []

        # generate a random number between 0 and 1
        # that represents a probability
        prob = random.random()

        if (prob < 0.25):
            # Make the record consists of two identical
            # but reversed sides
            #
            leftHalf = []
            for j in range(int(NUM_FEATURES/2)):
                leftHalf.append(random.randint(0,9))
            rightHalf = leftHalf[::-1]
            tmp = leftHalf + rightHalf

            # The label for this class is 0
            setAdd(records, tuple(tmp) + tuple([0]))

        elif (prob > 0.25 and prob < 0.5):
            # Make the record consists of two identical
            # but NOT reversed sides
            leftHalf = []
            for j in range(int(NUM_FEATURES/2)):
                leftHalf.append(random.randint(0,9))
            tmp = leftHalf + leftHalf

            # The label for this case should be 1
            setAdd(records, tuple(tmp) + tuple([1]))

        elif (prob > 0.5 and prob < 0.75):
            # Make the record consist of 4 quarters
            # Q1 : Q1-R : Q2 : Q2-R

            q1 = []
            q2 = []
            for j in range(int(NUM_FEATURES/4)):
                q1.append(random.randint(0,9))
                q2.append(random.randint(0,9))

            q1_r = q1[::-1]
            q2_r = q2[::-1]

            tmp = q1 + q1_r + q2 + q2_r

            # The label for this case should be 2
            setAdd(records, tuple(tmp) + tuple([2]))
            
        else:
            # Make the record consists of 4 quarters
            # Q1 : Q1 : Q2 : Q2
            q1 = []
            q2 = []
            for j in range(int(NUM_FEATURES/4)):
                q1.append(random.randint(0,9))
                q2.append(random.randint(0,9))

            tmp = q1 + q1 + q2 + q2

            # The label for this case should be 3
            setAdd(records, tuple(tmp) + tuple([3]))


    # sanity check
    assert(len(records) == NUM_RECORDS)

    for record in records:
        assert(len(record) == NUM_FEATURES+1)

    print('Everything looks good, saving dataset...')

    with open(DATASET_FILE+'.csv','w') as wf:
        for record in records:
            for i in range(len(record)-1):
                wf.write(f'{record[i]},')
            wf.write(f'{record[len(record)-1]}\n')

    print(f'Dataset file successfully written to {DATASET_FILE}.csv')
