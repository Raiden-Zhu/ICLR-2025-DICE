#!/bin/bash

# Define the path to the Python file to run
PYTHON_FILE="./main_new.py"

# Set hyperparameters
param1_range=(0) # choose node
param2_range=(0 1) # choose batch
node_datasize=50
batchsize=10
size=4
dataset_name=cifar10
mode=exponential
epochs=2
cuda=0
model=mlp
nonIID=false
choose_epoch=1
pretrained=0
image_size=28
log_file="./log/output__${size}_${dataset_name}_${mode}_${epochs}_${cuda}_${model}_${nonIID}_${choose_epoch}_pretrained${pretrained}.log"

# Run the Python file multiple times, each with different parameters
for param1 in "${param1_range[@]}"
do
    for param2 in "${param2_range[@]}"
    do
        echo "Running iteration with param1=$param1, param2=$param2, node_datasize=$node_datasize, batchsize=$batchsize, size=$size, dataset_name=$dataset_name, mode=$mode, epochs=$epochs, cuda=$cuda, model=$model, nonIID=$nonIID, choose_epoch=$choose_epoch, pretrained=$pretrained"
        python3 $PYTHON_FILE $param1 $param2 $node_datasize $batchsize $size $dataset_name $mode $epochs $cuda $model $nonIID $choose_epoch $pretrained $image_size > "$log_file" 2>&1  &
    done
done

wait
done
