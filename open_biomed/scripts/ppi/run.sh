#!/bin/bash
DEVICE="cuda:3"
MODEL="kedd"
DATASET="SHS148k"
SPLIT="dfs"

echo $MODEL-$DATASET-$SPLIT
for SEED in 43 44
do
python -u tasks/prot_task/ppi.py \
--device ${DEVICE} \
--mode train \
--config_path ./configs/ppi/${MODEL}.json \
--dataset ${DATASET} \
--dataset_path ../datasets/ppi/${DATASET} \
--split_strategy ${SPLIT} \
--output_path ../ckpts/finetune_ckpts/ppi-Sparse0805B/${MODEL}.pth \
--num_workers 8 \
--epochs 1000 \
--patience 200 \
--lr 1e-4 \
--logging_steps 15 \
--batch_size 512 \
--seed $SEED
done
