#!/bin/bash
py=python
DS_GEN=generate_dataset.py
MODEL=classify.py
PREDICT=useModel.py


# change into fc directory
cd fc

# Tryout different number of datasets
for NO_FEATURES in 4 8 12 16 20 24 
do
    # Remove existing model (if exists)
    rm -f model.pt || true

    NO_RECORDS=$(( NO_FEATURES * 100 ))

    # generate the dataset for the specified number of features
    $py $DS_GEN --features=$NO_FEATURES --records=$NO_RECORDS  


    DATAFILE=data-${NO_FEATURES}_${NO_RECORDS}.csv

    # train and save the model 
    $py $MODEL --datafile=$DATAFILE

    FEATURE_OUTFILE=predictions_${NO_FEATURES}.txt

    # Generate 100 random input strings and capture the results
    iterator=1
    while [ $iterator -le 100 ]
    do

        INPUT_PATTERN=$(cat /dev/urandom | tr -dc '0-9' | fold -w ${NO_FEATURES} | head -n 1)
        echo classifying input string $INPUT_PATTERN...
        echo `$py $PREDICT --input=${INPUT_PATTERN}` >> $FEATURE_OUTFILE
        ((iterator++))
    done

done



