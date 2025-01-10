#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
fabric run model pretrain.py --accelerator=cuda --devices=6 --devices 6 --train_data_dir data/openwebtext_processed/train --val_data_dir data/openwebtext_processed/validation --resume False --main-port 29300

