#!/bin/bash

# Define the path to the Python file to run
PYTHON_FILE="./main_new.py"

# Set hyperparameters
param1_range=(0) # choose node
param2_range=(4) # choose batch
node_datasize=500
batchsize=50
size=16
dataset_name=cifar10
mode=special0
epochs=10
cuda=0
model=resnet
nonIID=false
choose_epoch=5

log_file="./log/output_${dataset_name}_${mode}_${epochs}_${cuda}_${model}_${nonIID}_${choose_epoch}.log"

# Run the Python file multiple times, each with different parameters
for param1 in "${param1_range[@]}"
do
    for param2 in "${param2_range[@]}"
    do
        echo "Running iteration with param1=$param1, param2=$param2, node_datasize=$node_datasize, batchsize=$batchsize, size=$size, dataset_name=$dataset_name, mode=$mode, epochs=$epochs, cuda=$cuda, model=$model, nonIID=$nonIID, choose_epoch=$choose_epoch"
        python3 $PYTHON_FILE $param1 $param2 $node_datasize $batchsize $size $dataset_name $mode $epochs $cuda $model $nonIID $choose_epoch > "$log_file" 2>&1 &
    done
done

wait
done
