#!/bin/bash

# Define the path to the Python file to run
PYTHON_FILE="./main_new.py"
export https_proxy=http://9.131.113.25:11113

# Define parameter ranges
param1_range=(0 1) # choose node
param2_range=(0 1) # choose batch
node_datasize=100
batchsize=5
size=4
dataset_name=cifar100
mode=exponential
epochs=2
cuda=0
model=mlp
nonIID=false
image_size=28
pretrained=0
adam=false
log_file="./log/output_${size}_${dataset_name}__${node_datasize}_${mode}_${epochs}_${cuda}_${model}_${nonIID}_${pretrained}_adam${adam}.log"

# Run the Python file multiple times, each with different parameters
for param1 in "${param1_range[@]}"
do
    for param2 in "${param2_range[@]}"
    do
        echo "Running iteration with param1=$param1, param2=$param2, node_datasize=$node_datasize, batchsize=$batchsize, size=$size, dataset_name=$dataset_name, mode=$mode, epochs=$epochs, cuda=$cuda, model=$model, nonIID=$nonIID, pretrained=$pretrained, adam=$adam"
        python3 $PYTHON_FILE $param1 $param2 $node_datasize $batchsize $size $dataset_name $mode $epochs $cuda $model $nonIID $image_size $pretrained $adam > "$log_file" 2>&1 &
    done
done

wait
