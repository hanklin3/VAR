#!/bin/bash
#SBATCH -n 6 --gres=gpu:volta:2 -o output/train_debug.log-%j

source /etc/profile

source activate var_catk
module load anaconda/Python-ML-2024b

while true
do
    MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $MASTER_PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $MASTER_PORT

NGPUS=1
BATCH_SIZE=36
# d16, 256x256
# torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${MASTER_PORT} train.py \
#   --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --data_path=../data/imagenet/Data/CLS-LOC/

# torchrun --nproc_per_node=${NGPUS} --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=${MASTER_PORT} train.py \
#   --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --data_path=../data/imagenet/Data/CLS-LOC/

# torchrun --nproc_per_node=${NGPUS} --nnodes=1 --master_addr="localhost" --master_port=${MASTER_PORT} train.py \
#   --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --data_path=../data/imagenet/Data/CLS-LOC/ --bs=${BATCH_SIZE}

NGPUS=2
BATCH_SIZE=72
torchrun --nproc_per_node=${NGPUS} --nnodes=1 --master_addr="localhost" --master_port=${MASTER_PORT} train.py \
  --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1 --data_path=../data/imagenet/Data/CLS-LOC/ --bs=${BATCH_SIZE}


# # d16, 256x256
# torchrun \
#   -m \
#   --master-port $MASTER_PORT \
#   train \
#   depth=16 \
#   bs=768 \
#   ep=200 \
#   fp16=1 \
#   data_path="../data/imagenet/Data/CLS-LOC/" \
#   wpe=0.1 \
#   alng=1e-3
  
  