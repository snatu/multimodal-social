
#python3  robustness_analysis_siq.py -pretrain_config_file=../../pretrain/configs/base.yaml -ckpt=/data/raw/models/base.yaml/2022-12-03-22:17.17 -lr=5e-6 -ne=3 -output_grid_h=12 -output_grid_w=20 -mode=drop -percent=1 -modality=none -out_dir=robustness_no_audio
python3  robustness_analysis_siq.py -pretrain_config_file=../../pretrain/configs/base.yaml -ckpt=/data/raw/models/base.yaml/2022-12-03-22:17.17 -lr=5e-6 -ne=3 -output_grid_h=12 -output_grid_w=20 -mode=drop -percent=1 -modality=none -out_dir=robustness_no_audio

python3  robustness_analysis_siq.py -pretrain_config_file=../../pretrain/configs/base.yaml -ckpt=/data/raw/models/base.yaml/2022-12-04-15:42.07 -lr=5e-6 -ne=3 -output_grid_h=12 -output_grid_w=20 -mode=drop -percent=1 -modality=none -out_dir=robustness_no_audio_loss

python3  robustness_analysis_siq.py -pretrain_config_file=../../pretrain/configs/base.yaml -ckpt=/data/raw/models/base.yaml/2022-12-05-21:17.13 -lr=5e-6 -ne=3 -output_grid_h=12 -output_grid_w=20 -mode=noisy -percent=1 -modality=none -out_dir=robustness_only_audio

