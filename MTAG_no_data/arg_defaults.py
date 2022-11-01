defaults = [
    ("--optimizer", str, 'adamw'),
    ("--exclude_video", bool, False),
    ("--exclude_audio", bool, False),
    ("--exclude_text", bool, False),
    ("--bs", int, 15),
    ("--epochs", int, 50),
    ("--cuda", int, 0),
    ("--global_lr", float, 1e-3),
    ("--gru_lr", float, 1e-4),
    ("--beta1", float, 0.9),
    ("--beta2", float, 0.999),
    ("--eps", float, 1e-8),
    ('--weight_decay', float, 1e-2),
    ('--momentum', float, 0.9),
    ("--gnn_dropout", float, 0.1),
    ("--num_modality", int, 3),
    ("--num_frames", int, 50),
    ("--temporal_connectivity_order", int, 5),
    ("--num_video_aggr", int, 1),
    ("--num_text_aggr", int, 1),
    ("--num_audio_aggr", int, 1),
    ("--text_dim", int, 300),
    ("--audio_dim", int, 5),
    ("--video_dim", int, 20),
    ("--graph_conv_in_dim", int, 64),
    ("--graph_conv_out_dim", int, 512),
    ("--use_same_graph_in_out_dim", int, 0),
    ("--gat_conv_num_heads", int, 4),
    ("--useGNN", int, 1),
    ("--average_mha", int, 1),
    ("--num_gat_layers", int, 2),
    ("--lr_scheduler", str, 'reduce_on_plateau'),
    ("--reduce_on_plateau_lr_scheduler_patience", int, 5),
    ("--reduce_on_plateau_lr_scheduler_threshold", float, .002),
    ("--multi_step_lr_scheduler_milestones", str, None),
    ("--exponential_lr_scheduler_gamma", float, None),
    ("--use_pe", int, 1),
    ("--use_prune", int, 1),
    ("--prune_keep_p", float, 0.8),
    ("--use_ffn", int, 0),
    ("--graph_activation", str, None),
    ("--loss_type", str, "sl1"),
    ("--remove_isolated", int, 1),
    ("--use_conv1d", int, 0),
    ("--hidden_dim", int, 50),
    ("--net", str, "graphqa"),
    ("--test", int, 0),
    ("--graph_test", int, 0), # test with dummy graph
    
    ("--scene_mean", int, 1), # add mean node to scene rep
    # ("--scene_agg", int, 0), # add aggregator node to scene rep
    ("--social_baseline", int, 0),
    ("--factorized_key_subset", int, 1), # if using social baseline, whether to use full data or factorized_subset
    ("--importancegraph", int, 0), # add aggregator node to scene rep
    ("--num_agg_nodes", int, 2), # add aggregator node to scene rep
    ("--seq_len", int, 250), # max number of words per "utterance" in subgraph - no padding b/c graph NN
    ("--gran", str, 'chunk'), # granularity: chunk or word level alignment
    ("--align_pres_window", int, 1), # granularity: chunk or word level alignment
    ("--align_pastfut_window", int, 10), # granularity: chunk or word level alignment


    ("--text_feat", str, "glove"), # which features to use for text input in alignment
    ("--audio_feat", str, "covarep"),
    ("--video_feat", str, "densenet"),


    ("--use_loss_norm", int, 1),
    ("--use_all_to_all", int, 0),
    ("--checkpoints", str, '9,12,15,20'),
    ("--use_iemocap_inverse_sample_count_ce_loss", int, 0),
    ("--drop_1", float, 0.1),
    ("--drop_2", float, 0.1),
    ("--drop_het", float, 0.1),
    ("--early_stopping_patience", int, 10),

    # social-iq modeling
    ("--qa_strat", int, 0), # 0 q and a not part of graph; 1 connect q and a to all: feed contextual reps of q and a (self conns; rest of graph has been contextualized. Or not) through judge; 2 Find most important nodes through attentions, form qa graph w just them; 3 Stack these most important nodes in Temporal order; perform attention across this stack with query and answer as keys; feed outputs through judge;
    # masking out certain modalities
    ("--zero_out_video", int, 0),
    ("--zero_out_text",  int, 0),
    ("--zero_out_audio", int, 0),
    ("--flip_test_order", int, 0),
    ("--flip_train_order", int, 0),

    ("--out_dir", str, '/home/shounak_rtml/11777/MTAG/results/hash1'),
    ("--trials", int, 1),
    ("--train_block", int, 1),
    ("--test_block", int, 1),
    ("--use_ai_conn", int, 0),
    ("--use_qa_conn", int, 1),
    ("--use_qa_self_conn", int, 1),
    ("--use_mod_conn", int, 1),

    # Data augmentation for self-supervised contrastive learning

    ('--drop_nodes', int, 0),
    ('--permute_edges', int, 0),
    ('--subgraph', int, 0),
    ('--mask_nodes', int, 0),
    ('--drop_node_level', float, 0.2),
    ('--edge_perturb_level', float, 0.2),
    ('--subgraph_level', float, 0.2),
    ('--node_masking_level', float, 0.2),
    ('--temperature', float, 1.0),


        # Other settings, these are likely to be fixed all the time
    ("--log_dir", str, '/home/shounak_rtml/11777/MTAG/debug'),
    ("--eval", bool, False),
    ("--resume_pt", str, None),

    ("--single_gpu", bool, True),
    ("--load_model", bool, False),
    ("--pretrain_finetune", bool, False),
    ("--model_path", str, '/home/shounak_rtml/11777/MTAG/saved_model/model.pt'),
    ("--save_grad", bool, False),
    ("--dataset", str, "social"),
    ("--data_path", str, "/home/shounak_rtml/11777/MTAG/data"),
    ("--log_path", str, None),
    ("--padding_len", int, -1),
    ("--include_zero", bool, True),
        # for ablation
        # TODO：1. add flag to collapse past/present/future edge types into one type
        # TODO: 2. add flag gto collapse all 27 edge types into one type
        # TODO: 3. Add a flag to select random drop vs topk pruning
    ('--time_aware_edges', int, 1),
    ('--type_aware_edges', int, 1),
    ('--prune_type', str, 'topk'),

    ('--dummy', int, 0),
    ('--seed', int, 0),
    ('--return_layer_outputs', int, 0),
    ('--save_best_model', int, 0),
    ('--use_residual', int, 0),
    
]
