#/bin/bash!

# Run ViC-MAE/B-16 Finetuning on ImageNet
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/finetune_imagenet_vicmae_base.sh vit_base_patch16 vicmae_pretrain_vit_base_in1k_k400.pth imagenet/

OMP_NUM_THREADS=12 python -m torch.distributed.launch --nproc_per_node=8 image_finetune.py \
    --batch_size 128 \
    --accum_iter 1 \
    --epochs 90 \
    --model $1 \
    --finetune $2 \
    --epochs 100 \
    --blr 5e-4 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval \
    --data_path $3 \
    --log-wandb