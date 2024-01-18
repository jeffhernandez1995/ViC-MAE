import math
import sys
from typing import Iterable, Union

import torch
import numpy as np
import wandb
import util.misc as misc
import cv2
import util.lr_sched as lr_sched
# import matplotlib.pyplot as plt

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def denormalize(image):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    return np.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).astype(np.uint8)


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,
                    val_loader: Union[Iterable, None],
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # Take one sample from the data loader randomly
    # examples, _, = next(iter(data_loader))

    for data_iter_step, (image1, image2, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # # No plotting with norm_pix_loss
        # if data_iter_step % 500 == 0:
        #     if misc.get_rank() == 0:
        #         with torch.no_grad():
        #             # run MAE
        #             images = torch.clone(samples[:args.num_imgs_to_log]).to(device)
        #             loss, predictions, mask = model(images.float(), mask_ratio=args.mask_ratio)
        #             try:
        #                 predictions = model.module.unpatchify(predictions)
        #             except:
        #                 predictions = model.unpatchify(predictions)
        #             predictions = torch.einsum('nchw->nhwc', predictions).detach().cpu()

        #             # visualize the mask
        #             mask = mask.detach()
        #             try:
        #                 mask = (
        #                     mask
        #                     .unsqueeze(-1)
        #                     .repeat(1, 1, model.module.patch_embed.patch_size[0]**2 *3)
        #                 )  # (N, H*W, p*p*3)
        #                 mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
        #             except:
        #                 mask = (
        #                     mask
        #                     .unsqueeze(-1)
        #                     .repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)
        #                 )  # (N, H*W, p*p*3)
        #                 mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping                        
        #             mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        #             images = torch.einsum('nchw->nhwc', images).detach().cpu()

        #             # masked image
        #             im_masked = images * (1 - mask)

        #             # MAE reconstruction pasted with visible patches
        #             im_paste = images * (1 - mask) + predictions * mask
        #             images_to_plot = []
        #             for i in range(args.num_imgs_to_log):
        #                 grid = torch.cat([images[i], im_masked[i], predictions[i], im_paste[i]], dim=1)
        #                 images_to_plot.append(
        #                     wandb.Image(denormalize(grid.numpy()))
        #                 )
        #             wandb.log({'images': images_to_plot})
        #             del images, predictions, mask, im_masked, im_paste, grid, images_to_plot

        # samples = samples.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            loss, _, _ = model([image1, image2])
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()


        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)


        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            dataset_size = len(data_loader)
            epoch_1000x = int((data_iter_step / dataset_size + epoch) * 1000)
            dict_to_log = {
                'train_loss': loss_value_reduce,
                'lr': lr,
            }
            log_writer.log(
                dict_to_log,
                step=epoch_1000x,
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
