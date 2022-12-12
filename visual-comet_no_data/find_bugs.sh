count=50
for i in $(seq $count); do
    rm /home/shounak_rtml/11777/visual-comet/data/visualcomet/cache*
    python scripts/debug_run_generation.py --data_dir /home/shounak_rtml/11777/visual-comet/data/visualcomet/ --model_name_or_path experiments/image-inference/ --split val --overwrite_cache
    python scripts/update_val_annots.py
    python scripts/update_error_log.py
done

python scripts/run_generation.py --data_dir /home/shounak_rtml/11777/visual-comet/data/visualcomet/ --model_name_or_path experiments/image-inference/ --split val --overwrite_cache

