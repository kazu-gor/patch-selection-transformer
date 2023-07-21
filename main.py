import os
import time
import json
import random
import datetime
import argparse
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# from timm.utils import accuracy, AverageMeter
from timm.utils import AverageMeter

from lib.dataloader import get_loader
from lib.smooth_cross_entropy import SmoothCrossEntropy

# swin-transformer
from config import get_config
from models import build_model
from logger import create_logger
from lr_scheduler import build_scheduler
from utils import save_checkpoint, reduce_tensor, auto_resume_helper, NativeScalerWithGradNormCount


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')
    # iwasaki comment [2023-07-09 16:11:09]
    # path
    parser.add_argument('--train_path', type=str, default='../../dataset/TrainDataset/', help='path to train dataset')
    parser.add_argument('--val_path', type=str, default='../../dataset/ValDataset/', help='path to val dataset')
    # adam param
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args):
    # define dataset root path
    image_root = '{}/images/'.format(args.train_path)
    gt_root = '{}/masks/'.format(args.train_path)
    image_root_val = '{}/images/'.format(args.val_path)
    gt_root_val = '{}/masks/'.format(args.val_path)

    # define dataloader
    train_loader = get_loader(image_root,
                              gt_root,
                              shuffle=True,
                              batchsize=config.DATA.BATCH_SIZE,
                              trainsize=config.DATA.IMG_SIZE,
                              patch_size=config.MODEL.SWIN.PATCH_SIZE)
    val_loader = get_loader(image_root_val,
                            gt_root_val,
                            shuffle=False,
                            batchsize=1,
                            trainsize=config.DATA.IMG_SIZE,
                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                            phase='val')
    # dataloaders_dict = {"train": train_loader, "val": val_loader}

    # define model
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    loss_scaler = NativeScalerWithGradNormCount()
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    # define optimizer
    params = [p for p in model.parameters()]
    optimizer = torch.optim.Adam(
        [
            dict(params=params, lr=args.lr, betas=(args.beta1, args.beta2)),
        ],
    )

    # scheduler
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # define criterion
    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()
    # criterion = SmoothCrossEntropy(alpha=0.1)
    criterion = nn.BCEWithLogitsLoss()

    max_accuracy = 0.0

    # resume
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(msg)
        del checkpoint
        torch.cuda.empty_cache()

    # if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
    #     load_pretrained(config, model_without_ddp, logger)
    #     acc1, acc5, loss = validate(config, data_loader_val, model)
    #     logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
    # if config.THROUGHPUT_MODE:
    #     throughput(data_loader_val, model, logger)
    #     return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, criterion, train_loader,
                        optimizer, epoch, lr_scheduler, loss_scaler)
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger)

        # test the model
        acc1, acc2, acc3, loss = validate(config, val_loader, model)
        logger.info(f"Accuracy of the network on the {len(val_loader)} test images: {acc1:.1f}% {acc2:.1f}% {acc3:.1f}%")
        max_accuracy = max(max_accuracy, acc1, acc2, acc3)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    loss1_meter = AverageMeter()
    loss2_meter = AverageMeter()
    loss3_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, target_s1, target_s2, target_s3) in enumerate(data_loader):
        samples = samples.cuda()
        target_s1 = target_s1.cuda()
        target_s2 = target_s2.cuda()
        target_s3 = target_s3.cuda()

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)
        loss1 = criterion(outputs['stage1'], target_s1.float())
        loss2 = criterion(outputs['stage2'], target_s2.float())
        loss3 = criterion(outputs['stage3'], target_s3.float())
        loss = loss1 / config.TRAIN.ACCUMULATION_STEPS + \
            loss2 / config.TRAIN.ACCUMULATION_STEPS + \
            loss3 / config.TRAIN.ACCUMULATION_STEPS

        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), target_s1.size(0))
        loss1_meter.update(loss1.item(), target_s1.size(0))
        loss2_meter.update(loss2.item(), target_s1.size(0))
        loss3_meter.update(loss3.item(), target_s1.size(0))
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'loss1 {loss1_meter.val:.4f} ({loss1_meter.avg:.4f})\t'
                f'loss2 {loss2_meter.val:.4f} ({loss2_meter.avg:.4f})\t'
                f'loss3 {loss3_meter.val:.4f} ({loss3_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


def accuracy(output, target, threshold=0.5):
    assert output.size() == target.size(), 'size of output must be the same as target'
    preds = torch.sigmoid(output)
    pred_labels = torch.where(output > threshold, 1, 0)

    matching_elements = torch.eq(pred_labels, target)
    is_one = torch.eq(target, 1)

    correct = torch.logical_and(matching_elements, is_one).sum().item()
    wrong = torch.logical_xor(pred_labels, target).sum().item()

    with open('./output/validation/acc.txt', 'a') as f:
        if is_one.sum() == 0 and pred_labels.sum() == 0:
            f.write('acc: 100 %\n\n'
                    f'{pred_labels=}\n\n'
                    f'{target=}\n'
                    '--------------------------------------------------------------\n\n')
        else:
            f.write(f'acc: {correct / (correct + wrong) * 100} %\n\n'
                    f'{pred_labels=}\n\n'
                    f'{target=}\n'
                    '--------------------------------------------------------------\n\n')

    if is_one.sum() == 0 and pred_labels.sum() == 0:
        return 100.0
    return correct / (correct + wrong) * 100


@torch.no_grad()
def validate(config, data_loader, model):
    # criterion = SmoothCrossEntropy()
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc2_meter = AverageMeter()
    acc3_meter = AverageMeter()

    end = time.time()
    for idx, (samples, target_s1, target_s2, target_s3) in enumerate(data_loader):
        images = samples.cuda()
        target_s1 = target_s1.cuda()
        target_s2 = target_s2.cuda()
        target_s3 = target_s3.cuda()

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(images)

        # measure accuracy and record loss
        loss1 = criterion(outputs['stage1'], target_s1.float())
        loss2 = criterion(outputs['stage2'], target_s2.float())
        loss3 = criterion(outputs['stage3'], target_s3.float())
        loss = loss1 / config.TRAIN.ACCUMULATION_STEPS + \
            loss2 / config.TRAIN.ACCUMULATION_STEPS + \
            loss3 / config.TRAIN.ACCUMULATION_STEPS

        # acc1_stage1, acc5_stage1 = accuracy(outputs['stage1'], target_s1, topk=(1, 5))
        # acc1_stage2, acc5_stage2 = accuracy(outputs['stage2'], target_s2, topk=(1, 5))
        # acc1_stage3, acc5_stage3 = accuracy(outputs['stage3'], target_s3, topk=(1, 5))

        # acc1 = acc1_stage1 + acc1_stage2 + acc1_stage3
        # acc5 = acc5_stage1 + acc5_stage2 + acc5_stage3

        # acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        # loss = reduce_tensor(loss)

        acc1 = accuracy(outputs['stage1'], target_s1, threshold=0.6)
        acc2 = accuracy(outputs['stage2'], target_s2, threshold=0.6)
        acc3 = accuracy(outputs['stage3'], target_s3, threshold=0.6)

        loss_meter.update(loss.item(), target_s1.size(0))
        acc1_meter.update(acc1, target_s1.size(0))
        acc2_meter.update(acc2, target_s1.size(0))
        acc3_meter.update(acc3, target_s1.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@2 {acc2_meter.val:.3f} ({acc2_meter.avg:.3f})\t'
                f'Acc@3 {acc3_meter.val:.3f} ({acc3_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@2 {acc2_meter.avg:.3f} Acc@3 {acc3_meter.avg:.3f} ')
    return acc1_meter.avg, acc2_meter.avg, acc2_meter.avg, loss_meter.avg


if __name__ == "__main__":
    # DOSと勘違いされないようにする
    Image.MAX_IMAGE_PIXELS = 1000000000
    args, config = parse_option()

    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

#     # linear scale the learning rate according to total batch size, may not be optimal
#     linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
#     linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
#     linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
#     # gradient accumulation also need to scale the learning rate
#     if config.TRAIN.ACCUMULATION_STEPS > 1:
#         linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
#         linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
#         linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
#     config.defrost()
#     config.TRAIN.BASE_LR = linear_scaled_lr
#     config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
#     config.TRAIN.MIN_LR = linear_scaled_min_lr
#     config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    # logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    # if dist.get_rank() == 0:
    #     path = os.path.join(config.OUTPUT, "config.json")
    #     with open(path, "w") as f:
    #         f.write(config.dump())
    #     logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config, args)
