import os
os.environ["WANDB_PROJECT"] = "vicmae-pretrain"

import argparse
from copy import deepcopy
import datetime
import json
import numpy as np
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import timm
import torchvision.transforms as transforms
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vicmae
from engine_pretrain import train_one_epoch

import wandb
## FFCV dataset
from typing import List
import numpy as np
from ffcv.pipeline import PipelineSpec
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption

from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage, RandomColorJitter,RandomGrayscale, \
    RandomSolarization
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vicmae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.5, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--projector_hidden_dim', default=4096, type=int,
                        help='Projector hidden dimension.')
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='Temperature for contrastive loss.')
    parser.add_argument('--projector_out_dim', default=128, type=int,
                        help='Projector output dimension.')
    parser.add_argument('--weight_contrast', default=0.03, type=float,
                        help='Weight for contrastive loss.')
    parser.add_argument('--weight_recon', default=0.97, type=float,
                        help='Weight for reconstruction loss.')
    parser.add_argument('--norm_pix_loss', action='store_false',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=True)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    ## VideoFFCV dataset
    parser.add_argument('--data_path', default='imagenet_train.ffcv', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output/',
                        help='path where to tensorboard log')
    parser.add_argument('--log-wandb', action='store_true', default=False,
                        help='log training and validation metrics to wandb')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                        help='rank of distributed processes')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    args.device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    if args.output_dir is None:
        args.output_dir = 'output/'

    exp_name = '-'.join([
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model,
        str(args.input_size),
    ])
    args.output_dir = os.path.join(args.output_dir, exp_name)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.rank = misc.get_rank()

    if args.rank == 0 and args.log_wandb:
        logger = wandb.init(config=args)
    else:
        logger = None

    # # IMAGENET Data loading code
    train_decoder = RandomResizedCropRGBImageDecoder(
        (args.input_size, args.input_size),
    )
    image_pipeline_1: List[Operation] = [
        train_decoder,
        RandomHorizontalFlip(),
        RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
        RandomGrayscale(0.2),
        ToTensor(),
        ToDevice(torch.device(args.device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
    ]
    image_pipeline_2: List[Operation] = [
        train_decoder,
        RandomHorizontalFlip(),
        RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
        RandomGrayscale(0.2),
        RandomSolarization(0.2, 128),
        ToTensor(),
        ToDevice(torch.device(args.device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(args.device), non_blocking=True)
    ]

    order = OrderOption.RANDOM if args.distributed else OrderOption.QUASI_RANDOM
    data_loader_train = Loader(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        order=order,
        os_cache=True,
        drop_last=True,
        pipelines={
            'image1': image_pipeline_1,
            'image2': image_pipeline_2,
            'label': label_pipeline
        },
        distributed=args.distributed,
        seed=args.seed,
    )

    # define the model
    model = models_vicmae.__dict__[args.model](
        temperature=args.temperature,
        mask_ratio=args.mask_ratio,
        projector_hidden_dim=args.projector_hidden_dim,
        projector_out_dim=args.projector_out_dim,
        weight_contrast=args.weight_contrast,
        norm_pix_loss=args.norm_pix_loss,
    )

    model.to(args.device)

    args.num_imgs_to_log = args.batch_size if args.batch_size < 10 else 10

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler
    )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train, None,
            optimizer, args.device, epoch, loss_scaler,
            log_writer=logger,
            args=args
        )
        if args.output_dir and (epoch % 2 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
        }

        if args.output_dir and misc.is_main_process():
            # if log_writer is not None:
            #     log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
