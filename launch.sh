#!/bin/bash

source $HOME/.bashrc

SRC_DIR=${SRC_DIR:-/nfs/nfs3/users/kstachowicz/big_vision}
DATA_DIR=${DATA_DIR:-gs://rail-datasets-europe-west4/oxe/resize_256_256}
BATCH_SIZE=${BATCH_SIZE:-1024}

cd $SRC_DIR
mamba activate big_vision

python train_paligemma_bridge.py \
    --config.tokenizer_path models/paligemma_tokenizer.model \
    --config.model_path models/paligemma-3b-pt-224.f16.npz \
    --config.dataset_kwargs.oxe_kwargs.data_dir $DATA_DIR \
    --config.batch_size $BATCH_SIZE \
    --config.eval_interval 100 \
    --config.save_interval 1000 \
    --config.data_axis_size 1  --config.fsdp_axis_size -1 \
    --config.save_path "gs://kyle-checkpoints-eu4/paligemma-checkpoints"

read -p "Press any key to continue..."