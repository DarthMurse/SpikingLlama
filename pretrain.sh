#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
fabric run model pretrain.py --accelerator=cuda --devices=1 --devices 1 --train_data_dir data/openwebtext_processed/train --val_data_dir data/openwebtext_processed/validation --resume False --main-port 29300

