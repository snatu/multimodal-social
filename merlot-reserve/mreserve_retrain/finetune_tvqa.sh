#! /usr/bin/bash

cd finetune/tvqa
python3 tvqa_finetune.py ../../pretrain/configs/base.yaml  gs://merlotreserve/ckpts/base_resadapt -lr=5e-6 -ne=3 -scan_minibatch -output_grid_h=18 -output_grid_w=32
