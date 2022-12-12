#! /usr/bin/bash

# cd finetune/tvqa
python3 siq_finetune.py ../../pretrain/configs/base.yaml  ../../base.ckpt -lr=5e-6 -ne=10 -scan_minibatch -output_grid_h=18 -output_grid_w=32
