#!/bin/bash
module load gcc-10.2.0
# Pretrain
# SSL
bs=256
epoch=25
finetune_epochs=50
global_lr=0.001
finetune_lr=0.001
seq_len=25
temperature=0.01
mask_node=1
perturb_edge=1
drop_nodes=1
subgraph=1
drop_node_level=0.2
edge_perturb_level=0.2
subgraph_level=0.2
node_masking_level=0.2

model_name="model_bs_${bs}_epoch_${epoch}_lr_${global_lr}_seqlen_${seq_len}_temp_${temperature}_masknode_${mask_node}_${node_masking_level}_perturbedge_${perturb_edge}_${edge_perturb_level}_dropnodes_${drop_nodes}_${drop_node_level}_subgraph_${subgraph}_${subgraph_level}.pt"
echo ${model_name}
echo "finetune_epoch_${finetune_epochs}_finetune_lr_${finetune_lr}_${model_name}.txt"
if test -f "/home/shounak_rtml/11777/MTAG/saved_model/${model_name}"; then
    echo "configuration already trained"
    echo "Pretraining finished. Starting fine-tuning with learning rate ${finetune_lr} ... "
    # Fine-tune
    # Supervised
    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --bs 3 \
    --drop_het 0 \
    --epochs $finetune_epochs \
    --gat_conv_num_heads 2 \
    --global_lr $finetune_lr \
    --graph_conv_in_dim 80 \
    --net factorized \
    --num_agg_nodes 1 \
    --num_gat_layers 2 \
    --scene_mean 1 \
    --social_baseline 0 \
    --out_dir /home/shounak_rtml/11777/MTAG/results/factorized \
    --data_path /home/shounak_rtml/11777/MTAG/data/ \
    --model_path "/home/shounak_rtml/11777/MTAG/saved_model/${model_name}" \
    --trials 1 \
    --seq_len $seq_len \
    --dataset social \
    --gran word \
    --test 0 \
    --zero_out_video 0 \
    --zero_out_audio 0 \
    --zero_out_text 0 \
    --pretrain_finetune true >> "results/finetune_epoch_${finetune_epochs}_finetune_lr_${finetune_lr}_${model_name}.txt"
else
    CUDA_VISIBLE_DEVICES=0 python main_ssl.py \
    --bs $bs \
    --drop_het 0 \
    --epochs $epoch \
    --gat_conv_num_heads 2 \
    --global_lr $global_lr \
    --graph_conv_in_dim 80 \
    --net factorized \
    --num_agg_nodes 1 \
    --num_gat_layers 2 \
    --scene_mean 1 \
    --social_baseline 0 \
    --out_dir /home/shounak_rtml/11777/MTAG/results/factorized \
    --data_path /home/shounak_rtml/11777/MTAG/data/ \
    --model_path "/home/shounak_rtml/11777/MTAG/saved_model/${model_name}" \
    --trials 1 \
    --seq_len $seq_len \
    --dataset social \
    --gran word \
    --test 0 \
    --zero_out_video 0 \
    --zero_out_audio 0 \
    --zero_out_text 0 \
    --permute_edges $perturb_edge \
    --mask_nodes $mask_node \
    --subgraph $subgraph \
    --drop_nodes $drop_nodes \
    --temperature $temperature \
    --drop_node_level $drop_node_level \
    --edge_perturb_level $edge_perturb_level \
    --subgraph_level $subgraph_level \
    --node_masking_level $node_masking_level
    
    echo "Pretraining finished. Starting fine-tuning with learning rate ${finetune_lr} ... "
    # Fine-tune
    # Supervised
    CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --bs 3 \
    --drop_het 0 \
    --epochs $finetune_epochs \
    --gat_conv_num_heads 2 \
    --global_lr $finetune_lr \
    --graph_conv_in_dim 80 \
    --net factorized \
    --num_agg_nodes 1 \
    --num_gat_layers 2 \
    --scene_mean 1 \
    --social_baseline 0 \
    --out_dir /home/shounak_rtml/11777/MTAG/results/factorized \
    --data_path /home/shounak_rtml/11777/MTAG/data/ \
    --model_path "/home/shounak_rtml/11777/MTAG/saved_model/${model_name}" \
    --trials 1 \
    --seq_len $seq_len \
    --dataset social \
    --gran word \
    --test 0 \
    --zero_out_video 0 \
    --zero_out_audio 0 \
    --zero_out_text 0 \
    --pretrain_finetune true >> "results/finetune_epoch_${finetune_epochs}_finetune_lr_${finetune_lr}_${model_name}.txt"
fi
