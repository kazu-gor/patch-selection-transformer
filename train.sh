CUDA_LAUNCH_BLOCKING=1 PYTHONPATH="./swin-transformer/:$PYTHONPATH" python3 -m torch.distributed.launch main.py --cfg ./swin-transformer/configs/pst/swin_base_patch4_window12_352_finetune_patch_selection.yaml --local_rank 0 --output output/pst/default
