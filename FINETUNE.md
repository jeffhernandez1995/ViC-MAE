## Fine-tuning Pre-trained MAE for Classification

### Fine-tuning ViC-MAE/B-16 on ImageNet-1K

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/finetune_imagenet_vicmae_base.sh vit_base_patch16 vicmae_pretrain_vit_base_in1k_k400.pth imagenet/
```

### Fine-tuning ViC-MAE/L-16 on ImageNet-1K

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/finetune_imagenet_vicmae_large.sh vit_large_patch16 vicmae_pretrain_vit_large_in1k_k400.pth imagenet/
```

### Fine-tuning ViC-MAE/B-16 on Kinetics-400

Coming soon.

### Fine-tuning ViC-MAE/L-16 on Kinetics-400

Coming soon.

#### Notes
The [pre-trained models we provide](README.md#checkpoints) are trained with *normalized* pixels `--norm_pix_loss` (800 epochs). The fine-tuning hyper-parameters are slightly different from the default baseline using *unnormalized* pixels.