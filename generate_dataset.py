##
##  Author: Khaled Abdelaal
##          khaled.abdelaal@ou.edu
##
##          A simple script to generate a dataset of NUM_FEATURES features
##              and NUM_RECORDS records with some simple patterns in each
##              of the generated records
##
###############################################################

import random

NUM_FEATURES = 12 # For now, it must be a multiple of 4
NUM_RECORDS = 1200 # can be any number <= 2^12
DATASET_FILE='data' # dataset filename


# This is where the class labels are saved
output = []

# Stores input instances
records = []

# generate random input
for i in range(NUM_RECORDS):
    records.append([])

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
        records[i] = leftHalf + rightHalf

        # The label for this case should be 0
        output.append(0)

    elif (prob > 0.25 and prob < 0.5):
        # Make the record consists of two identical
        # but NOT reversed sides
        leftHalf = []
        for j in range(int(NUM_FEATURES/2)):
            leftHalf.append(random.randint(0,9))
        records[i] = leftHalf + leftHalf

        # The label for this case should be 1
        output.append(1)

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

        records[i] = q1 + q1_r + q2 + q2_r

        # The label for this case should be 2
        output.append(2)
    else:
        # Make the record consists of 4 quarters
        # Q1 : Q1 : Q2 : Q2
        q1 = []
        q2 = []
        for j in range(int(NUM_FEATURES/4)):
            q1.append(random.randint(0,9))
            q2.append(random.randint(0,9))

        records[i] = q1 + q1 + q2 + q2

        # The label for this case should be 3
        output.append(3)


# sanity check
assert(len(records) == NUM_RECORDS)
assert(len(records[0]) == NUM_FEATURES)
assert(len(output) == NUM_RECORDS)

print('Everything looks good, saving dataset...')

with open(DATASET_FILE+'.csv','w') as wf:
    for i in range(len(records)):
        for j in range(len(records[i])):
            wf.write(f'{records[i][j]},')
        #wf.write('\n')
        wf.write(f'{output[i]}\n')

print(f'Dataset file successfully written to {DATASET_FILE}.csv')
