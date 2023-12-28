#!/bin/bash
MODE="train"
MODEL="molfm"
DEVICE=0,1,2,3,4,5,6,7
EPOCHS=100
SPLIT="warm"

#echo "Training on MSSL2drug's dataset, split is "${SPLIT}
CUDA_VISIBLE_DEVICES=${DEVICE} python tasks/mol_task/ddi.py \
--device cuda:0 \
--config_path ./configs/ddi/${MODEL}.json \
--dataset mssl2drug \
--dataset_path ../datasets/drugbankddi \
--output_path ../ckpts/finetune_ckpts/ddi/${MODEL}.pth \
--mode ${MODE} \
--epochs ${EPOCHS} \
--num_workers 1 \
--batch_size 512 \
--lr 1e-4 \
--logging_steps 50 \
--patience 50