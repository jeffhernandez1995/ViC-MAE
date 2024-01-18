## Pre-training MAE
To pre-train ViT-Large (recommended default), run the following:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=12 python -m torch.distributed.launch --nproc_per_node=8 pretrain.py \
    --batch_size 128 \
    --accum_iter 4 \
    --epochs 801 \
    --model vicmae_vit_base_patch16 \
    --input_size 224 \
    --mask_ratio 0.5 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --output_dir output/ \
    --projector_hidden_dim 4096 \
    --projector_out_dim 128 \
    --weight_contrast 0.03 \
    --weight_recon 0.97 \
    --temperature 0.1 \
    --num_workers 50 \
    --data_path in1k_k400_train.ffcv \
    --log-wandb
```
Here we use --norm_pix_loss (True as default) as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off --norm_pix_loss. To train ViT-Base set --model vicmae_base_patch16.

### Notes
To get `in1k_k400_train.ffcv`, see [Data Preparation](DATA.md#data-preparation).