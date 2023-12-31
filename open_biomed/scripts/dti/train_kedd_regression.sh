#!/bin/bash
MODE="train"
MODEL="kedd"
BASE="graphmvp"
DEVICE="cuda:0"
EPOCHS=1000

python tasks/mol_task/dti.py \
--device ${DEVICE} \
--config_path ./configs/dti/${MODEL}-${BASE}.json \
--dataset davis \
--dataset_path ../datasets/dti/davis \
--output_path ../ckpts/finetune_ckpts/dti/${MODEL}.pth \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 4 \
--batch_size 128 \
--lr 1e-3 \
--logging_steps 500 \
--patience 1000
