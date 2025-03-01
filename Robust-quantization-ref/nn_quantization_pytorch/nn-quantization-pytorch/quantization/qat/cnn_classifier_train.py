import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import random
import shutil
import time
import datetime
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import numpy as np
from utils.data import get_dataset
from utils.preprocess import get_transform
from quantization.quantizer import ModelQuantizer, OptimizerBridge
from pathlib import Path
from utils.mllog import MLlogger
from utils.meters import AverageMeter, ProgressMeter, accuracy
from torch.optim.lr_scheduler import StepLR
from models.resnet import resnet as custom_resnet
from models.inception import inception_v3 as custom_inception
from quantization.qat.module_wrapper import ActivationModuleWrapper, ParameterModuleWrapper
from utils.misc import normalize_module_name, arch2depth
from models.mlp_cifar10 import cifar10_mlp

log_data = {
    "experiment": None,
    "logs": []
}

home = str(Path.home())
models.mlp_cifar10 = cifar10_mlp()
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
#print('*********************',model_names) 

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--datapath', metavar='DATAPATH', type=str, default=None,
                    help='dataset folder')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-ep', '--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_step', '--learning-rate-step', default=25, type=int,
                    help='learning rate reduction step')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--pretrained', type=str, default=None, 
                    help='Path to the pre-trained model file')
parser.add_argument('--custom_resnet', action='store_true', help='use custom resnet implementation')
parser.add_argument('--custom_inception', action='store_true', help='use custom inception implementation')
parser.add_argument('--custom_mlp', action='store_true', help='use custom mlp implementation')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu_ids', default=[0], type=int, nargs='+',
                    help='GPU ids to use (e.g 0 1 2 3)')
parser.add_argument('--lr_freeze', action='store_true', help='Freeze learning rate', default=False)
parser.add_argument('--bn_folding', '-bnf', action='store_true', help='Apply Batch Norm folding', default=False)
parser.add_argument('--log_stats', '-ls', action='store_true', help='Log statistics', default=False)

parser.add_argument('--quantize', '-q', action='store_true', help='Enable quantization', default=False)
parser.add_argument('--experiment', '-exp', help='Name of the experiment', default='default')
parser.add_argument('--bit_weights', '-bw', type=int, help='Number of bits for weights', default=None)
parser.add_argument('--bit_act', '-ba', type=int, help='Number of bits for activations', default=None)
parser.add_argument('--model_freeze', '-mf', action='store_true', help='Freeze model parameters', default=False)
parser.add_argument('--temperature', '-t', type=float, help='Temperature parameter for sigmoid quantization', default=None)
parser.add_argument('--qtype', default='None', help='Type of quantization method')
parser.add_argument('--bcorr_w', '-bcw', action='store_true', help='Bias correction for weights', default=False)
parser.add_argument('--stochastic', '-sr', action='store_true', help='stochastic rounding', default=False)

best_acc1 = 0

import json
def save_log_to_json(log_dir, log_data):
    """Lưu log vào file JSON."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.json")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=4)


def main():
    args = parser.parse_args()
    # args.seed = None # temp moran
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    curr_proj_dir = Path.cwd() / "mxt" / "logs"
    log_data["experiment"] = args.experiment
    log_dir = curr_proj_dir / args.experiment
    os.makedirs(log_dir, exist_ok=True)

    print(f"Logging directory: {log_dir}")
    main_worker(args, log_dir)

def main_worker(args, log_dir):
    global best_acc1
    log_data = {"experiment": args.experiment, "logs": []}
    log_file = os.path.join(log_dir, "training_log.json")

    datatime_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    suf_name = "_" + args.experiment

    if args.gpu_ids is not None:
        print("Use GPU: {} for training".format(args.gpu_ids))

    if args.log_stats:
        from utils.stats_trucker import StatsTrucker as ST
        ST("W{}A{}".format(args.bit_weights, args.bit_act))

    if 'resnet' in args.arch and args.custom_resnet:
        # pdb.set_trace()
        model = custom_resnet(arch=args.arch, pretrained=args.pretrained, depth=arch2depth(args.arch), dataset=args.dataset)
    elif 'inception_v3' in args.arch and args.custom_inception:
        model = custom_inception(pretrained=args.pretrained)
    elif 'mlp_cifar10' in args.arch:
        model = cifar10_mlp(pretrained=args.pretrained, weight_path=args.pretrained)

    device = torch.device('cuda:{}'.format(args.gpu_ids[0]))
    cudnn.benchmark = True

    torch.cuda.set_device(args.gpu_ids[0])
    model = model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, device)
            args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            # best_acc1 may be from a checkpoint from a different GPU
            # best_acc1 = best_acc1.to(device)
            checkpoint['state_dict'] = {normalize_module_name(k): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if len(args.gpu_ids) > 1:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features, args.gpu_ids)
        else:
            model = torch.nn.DataParallel(model, args.gpu_ids)

    default_transform = {
        'train': get_transform(args.dataset, augment=True),
        'eval': get_transform(args.dataset, augment=False)
    }

    val_data = get_dataset(args.dataset, 'val', default_transform['eval'])
    # val_loader = torch.utils.data.DataLoader(
    #     val_data,
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=100,
    shuffle=False,
    num_workers=2,
    pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    train_data = get_dataset(args.dataset, 'train', default_transform['train'])
    #train_loader = torch.utils.data.DataLoader(
        # train_data,
        # batch_size=args.batch_size, shuffle=True,
        # num_workers=args.workers, pin_memory=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=200,  
    shuffle=True,
    num_workers=2,  
    pin_memory=True,
    drop_last=True)

    # TODO: replace this call by initialization on small subset of training data
    # TODO: enable for activations
    # validate(val_loader, model, criterion, args, device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)


    # Initialize log file for quantizer state and metrics
    quantizer_log_file = os.path.join(log_dir, "quantizer_log.json")
    metrics_log_file = os.path.join(log_dir, "metrics_log.json")

    if not os.path.exists(quantizer_log_file):
        with open(quantizer_log_file, "w") as f:
            json.dump({}, f)

    if not os.path.exists(metrics_log_file):
        with open(metrics_log_file, "w") as f:
            json.dump({}, f)

    # Modified code
    mq = None
    if args.quantize:
        if args.bn_folding:
            print("Applying batch-norm folding ahead of post-training quantization")
            from utils.absorb_bn import search_absorbe_bn
            search_absorbe_bn(model)

        all_relu = [n for n, m in model.named_modules() if isinstance(m, nn.ReLU)]
        all_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

        layers = all_relu[1:-1] + all_linear[1:]

        replacement_factory = {
            nn.ReLU: ActivationModuleWrapper,
            nn.Linear: ParameterModuleWrapper
        }
        mq = ModelQuantizer(
            model, args, layers, replacement_factory,
            OptimizerBridge(optimizer, settings={'algo': 'SGD', 'dataset': args.dataset})
        )

        if args.resume:
            # Load quantization parameters from state dict
            mq.load_state_dict(checkpoint['state_dict'])

        # Save quantizer state to JSON
        quantizer_state = mq.get_quantizer_state()
        with open(quantizer_log_file, "w") as f:
            json.dump(quantizer_state, f, indent=4)

        if args.model_freeze:
            mq.freeze()

    # Validation or evaluation logging
    if args.evaluate:
        acc = validate(val_loader, model, criterion, args, device)

        # Save metrics to JSON
        with open(metrics_log_file, "r") as f:
            metrics = json.load(f)

        metrics["Val Acc1"] = acc
        with open(metrics_log_file, "w") as f:
            json.dump(metrics, f, indent=4)



    # Evaluate on validation set
    acc1 = validate(val_loader, model, criterion, args, device)

    # Save metrics to JSON
    with open(metrics_log_file, "r") as f:
        metrics = json.load(f)

    metrics["Val Acc1"] = acc1
    with open(metrics_log_file, "w") as f:
        json.dump(metrics, f, indent=4)


    for epoch in range(0, args.epochs):
        # train for one epoch
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print('Timestamp Start epoch: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
        # Training step
        train_loss, train_acc = train(
            train_loader, model, criterion, optimizer, epoch, args, device, log_file=os.path.join(log_dir, "training_log.json"), mq=mq)
        print('Timestamp End epoch: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

        if not args.lr_freeze:
            lr_scheduler.step()

        # evaluate on validation set
        val_acc = validate(val_loader, model, criterion, args, device)
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }
        log_data["logs"].append(log_entry)

        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=4)

        # Update learning rate
        if not args.lr_freeze:
            lr_scheduler.step()

        # Evaluate with k-means quantization if enabled
        if args.model_freeze and args.quantize:
            with mq.quantization_method("kmeans"):
                kmeans_acc = validate(val_loader, model, criterion, args, device)
                print(f"K-means validation accuracy: {kmeans_acc:.2f}")
                log_entry["kmeans_acc"] = kmeans_acc

        # Log quantizer state if quantization is enabled
        if args.quantize:
            quantizer_state = mq.get_quantizer_state()
            log_entry["quantizer_state"] = quantizer_state
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=4)

        # remember best acc@1 and save checkpoint
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)
        print(f"Best accuracy so far: {best_acc1:.2f}")
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict() if len(args.gpu_ids) == 1 else model.module.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, datatime_str=datatime_str, suf_name=suf_name)


def train(train_loader, model, criterion, optimizer, epoch, args, device, log_file, mq=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    best_acc1 = -1
    end = time.time()
    epoch_log = {"epoch": epoch + 1, "batches": []}
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # Add batch metrics to log
        batch_log = {
            "batch": i + 1,
            "loss": loss.item(),
            "train_acc1": acc1.item(),
            "train_acc5": acc5.item(),
        }
        epoch_log["batches"].append(batch_log)

        if i % args.print_freq == 0:
            progress.print(i)

    # Save epoch metrics to the log file
    with open(log_file, "a") as f:
        json.dump(epoch_log, f, indent=4)

    return losses.avg, top1.avg



def validate(val_loader, model, criterion, args, device):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='last_checkpoint.pth.tar', datatime_str='', suf_name=''):
    if datatime_str == '':
        print("no datatime_str")
        exit(1)
    ckpt_dir = os.path.join(os.getcwd(), 'mxt-sim', 'ckpt', state['arch'], datatime_str)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, filename)
    torch.save(state, ckpt_path)
    print("ckpt dir: " + str(ckpt_path))
    if is_best:
        shutil.copyfile(ckpt_path, os.path.join(ckpt_dir, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
