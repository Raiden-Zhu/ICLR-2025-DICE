#!/bin/bash

# 定义要运行的 Python 文件路径
PYTHON_FILE="/mnt/csp/mmvision/home/lwh/DLS_2/main_new.py"

# 定义参数范围
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

log_file="/mnt/csp/mmvision/home/lwh/DLS_2/log/output_${dataset_name}_${mode}_${epochs}_${cuda}_${model}_${nonIID}_${choose_epoch}.log"

# 运行 Python 文件十次，每次传递不同的参数
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