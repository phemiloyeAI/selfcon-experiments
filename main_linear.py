from __future__ import print_function

import os
import sys
import argparse
import warnings
import time
import math
import random
import uuid
import wandb
import yaml
import builtins
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision import transforms, datasets

from utils.util import *
from utils.datasets import get_set

from networks.resnet_big import ConResNet, LinearClassifier
from networks.vgg_big import ConVGG, LinearClassifier_VGG
from networks.wrn_big import ConWRN, LinearClassifier_WRN
from networks.efficient_big import ConEfficientNet, LinearClassifier_EFF


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--project_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./save/representation')
    parser.add_argument('--ckpt', type=str, default='',
                        help='path for pre-trained model')
    parser.add_argument('--subnet', action='store_true',
                        help='measure the accuracy of sub-network or not')

    # dataset
    parser.add_argument('--dataset_name', type=str, default='imagenet', choices=[
        'cifar10', 'flowers', 'cifar100', 
        'tinyimagenet', 'imagenet', 'imagenet100', 
        'cub', 'aircraft', 'cars'])
    
    parser.add_argument('--data_cfg', type=str, default='configs/datasets/flowers.yml')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=12)

    parser.add_argument('--dataset_root_path', type=str, default='../../data/flowers/', help='Path to the training.')
    parser.add_argument('--train_trainval', action='store_true', help='True to use train_val split for training')
    parser.add_argument('--df_train', type=str, default='train.csv', help='Path to the training dataset CSV file.')
    parser.add_argument('--df_trainval', type=str, default='train_val.csv', help='Path to the train-validation dataset CSV file.')
    parser.add_argument('--df_val', type=str, default='val.csv', help='Path to the validation dataset CSV file.')
    parser.add_argument('--df_test', type=str, default='test.csv', help='Path to the test dataset CSV file.')

    parser.add_argument('--folder_train', type=str, default='jpg', help='Folder path for training images.')
    parser.add_argument('--folder_val', type=str, default='jpg', help='Folder path for validation images.')
    parser.add_argument('--folder_test', type=str, default='jpg', help='Folder path for test images.')


    # model 
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--selfcon_pos', type=str, default='[False,False,False]', 
                        help='where to augment the paths')
    parser.add_argument('--selfcon_arch', type=str, default='resnet', 
                        choices=['resnet', 'vgg', 'efficientnet', 'wrn'], help='which architecture to form a sub-network')
    parser.add_argument('--selfcon_size', type=str, default='same', 
                        choices=['fc', 'same', 'small'], help='argument for num_blocks of a sub-network')

    # optimization
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset_name, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset_name == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset_name == 'cub':
      opt.n_cls = 200 
    elif opt.dataset_name == 'aircraft':
      opt.n_cls = 200
    elif opt.dataset_name == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset_name == 'tinyimagenet':
        opt.n_cls = 200
    elif opt.dataset_name == 'imagenet':
        opt.n_cls = 1000
    elif opt.dataset_name == 'imagenet100':
        opt.n_cls = 100
    elif opt.dataset_name == 'flowers':
        opt.n_cls = 102
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset_name))

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif opt.dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32

    elif opt.dataset_name in ['cub', 'aircraft', 'car', 'flowers']:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 224
    elif opt.dataset_name == 'tinyimagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 64
    elif opt.dataset_name == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 224
    elif opt.dataset_name == 'imagenet100':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 224
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset_name))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        normalize
    ])

    if opt.dataset_name in ['imagenet', 'imagenet100']:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    else:
        val_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize])

    train_dataset = get_set(opt, split='train', transform=train_transform)
    val_dataset = get_set(opt, split='val', transform=val_transform)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    return train_loader, val_loader, train_sampler

def set_model(opt):
    model_kwargs = {'name': opt.model, 
                    'dataset': opt.dataset_name,
                    'selfcon_pos': eval(opt.selfcon_pos),
                    'selfcon_arch': opt.selfcon_arch,
                    'selfcon_size': opt.selfcon_size
                    }
    
    if opt.model.startswith('resnet'):
        model = ConResNet(**model_kwargs)
        print(model)
        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
        if opt.subnet:
            sub_classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)
            
    elif opt.model.startswith('vgg'):
        model = ConVGG(**model_kwargs)
        classifier = LinearClassifier_VGG(name=opt.model, num_classes=opt.n_cls)
        if opt.subnet:
            sub_classifier = LinearClassifier_VGG(name=opt.model, num_classes=opt.n_cls)
            
    elif opt.model.startswith('wrn'):
        model = ConWRN(**model_kwargs)
        classifier = LinearClassifier_WRN(name=opt.model, num_classes=opt.n_cls)
        if opt.subnet:
            sub_classifier = LinearClassifier_WRN(name=opt.model, num_classes=opt.n_cls)

    elif opt.model.startswith('eff'):
        model = ConEfficientNet(**model_kwargs)
        classifier = LinearClassifier_EFF(name=opt.model, num_classes=opt.n_cls)
        if opt.subnet:
            sub_classifier = LinearClassifier_EFF(name=opt.model, num_classes=opt.n_cls)

    print('Linear Classifier: ', classifier)

    criterion = torch.nn.CrossEntropyLoss()
    if opt.ckpt:
        ckpt = torch.load(opt.ckpt, map_location='cpu')
        print()
        # print('SelfCon Layer: ', ckpt['encoder.selfcon_layer'])
        state_dict = ckpt['model']
        # state_dict['encoder.conv1.weight'] = torch.nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=1)

        # print(ckpt['model']['encoder.conv1.weight'].shape)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            if opt.ckpt:
                new_state_dict = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "")
                    new_state_dict[k] = v
                state_dict = new_state_dict
        
        model.cuda()
        classifier = classifier.cuda()
        if opt.subnet:
            sub_classifier = sub_classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        if opt.ckpt:
            state_dict = {k.replace("downsample", "shortcut"): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)

    if not opt.subnet:
        sub_classifier = None
    return model, classifier, sub_classifier, criterion, opt


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt, subnet=False):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
            # print('Features: ', features)
            features = features[-1] if not subnet else features[0][-1]
            
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, sub_classifier, criterion, opt, best_acc):
    def __update_metric(output, labels, top1, top5, bsz):
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)
        top5.update(acc5[0], bsz)
        
        return top1, top5        
        
    def __best_acc(val_acc1, val_acc5, best_acc, key='backbone'):
        if val_acc1.item() > best_acc[key][0]:
            best_acc[key][0] = val_acc1.item()
            best_acc[key][1] = val_acc5.item()
            
        return best_acc
    
    """validation"""
    model.eval()
    classifier.eval()
    if sub_classifier:
        sub_classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = AverageMeter(), AverageMeter()
    top1_sub, top5_sub = AverageMeter(), AverageMeter()
    top1_ens, top5_ens = AverageMeter(), AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            features = model.encoder(images)
            output = classifier(features[-1])
            loss = criterion(output, labels)
            
            # for only one subnetwork
            if opt.subnet:
                sub_output = sub_classifier(features[0][-1])
                ensemble_output = (output + sub_output) / 2

            # update metric
            losses.update(loss.item(), bsz)
            top1, top5 = __update_metric(output, labels, top1, top5, bsz)
            if opt.subnet:
                top1_sub, top5_sub = __update_metric(sub_output, labels, top1_sub, top5_sub, bsz)
                top1_ens, top5_ens = __update_metric(ensemble_output, labels, top1_ens, top5_ens, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.2f}, Acc@5 {top5.avg:.2f}'.format(top1=top1, top5=top5))
    best_acc = __best_acc(top1.avg, top5.avg, best_acc)
    
    if opt.subnet:
        print(' * Acc@1 {top1.avg:.2f}, Acc@5 {top5.avg:.2f}'.format(top1=top1_sub, top5=top5_sub))
        best_acc = __best_acc(top1_sub.avg, top5_sub.avg, best_acc, key='sub')
        
        print(' * Acc@1 {top1.avg:.2f}, Acc@5 {top5.avg:.2f}'.format(top1=top1_ens, top5=top5_ens))
        best_acc = __best_acc(top1_ens.avg, top5_ens.avg, best_acc, key='ensemble')           
    return best_acc


def main():
    opt = parse_option()

    data_config = yaml.safe_load(open(opt.data_cfg, 'r'))
    for key, value in data_config.items():
        if hasattr(opt, key):
            setattr(opt, key, value)

    wandb.init(
        
        project=opt.project_name,
        name=opt.model_name,
        id=f'{str(uuid.uuid4())[:5]}',
        config = opt
    )
    
    # fix seed
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    cudnn.deterministic = True
    
    best_acc = {'backbone': [0, 0, 0],
                'sub': [0, 0, 0],
                'ensemble': [0, 0]}
    
    # build model and criterion
    model, classifier, sub_classifier, criterion, opt = set_model(opt)

    # build data loader
    train_loader, val_loader, train_sampler = set_loader(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)
    sub_optimizer = set_optimizer(opt, sub_classifier) if opt.subnet else None
    
    start_time = time.time()
    
    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        if opt.subnet:
            adjust_learning_rate(opt, sub_optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        best_acc['backbone'][2] = acc.item()
        
        if opt.subnet:
            _, sub_acc = train(train_loader, model, sub_classifier, criterion,
                              sub_optimizer, epoch, opt, subnet=True)
            print('Train epoch {}, accuracy:{:.2f}'.format(
                epoch, sub_acc))
            best_acc['sub'][2] = sub_acc.item()

        # eval for one epoch
        best_acc = validate(val_loader, model, classifier, sub_classifier, criterion, opt, best_acc)

        wandb.log({

           "train_loss": loss,
            "val_top1_backbone": best_acc["backbone"][0],
            "val_top5_backbone": best_acc["backbone"][1],
            "val_top1_subnet": best_acc["sub"][0],
            "val_top5_subnet": best_acc["backbone"][1],
            "val_top1_ensemble": best_acc["ensemble"][0],
            "val_top5_ensemble": best_acc["ensemble"][1],
            "train_acc_backbone": best_acc["backbone"][2],
            "train_acc_subnet": best_acc["sub"][2],
        })
    
    time_total = time.time() - start_time
    time_total = round(time_total / 60, 2) ## Total time taken to train the model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

    ## Number of model parameters
    no_params = sum([p.numel() for p in model.parameters()])
    no_params = round(no_params / (1e6), 2)  # millions of parameters

    ## Totatl Memory Used                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
    max_memory = round(max_memory, 2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    #save_checkpoint(model.state_dict(), False, filename=file_name)

    wandb.run.summary["total_training_time"] = time_total
    wandb.run.summary["number_of_model_params"] = no_params
    wandb.run.summary["maximum_memory_reserved"] = max_memory
            
    update_json(opt.ckpt + '_%s' % opt.exp_name if opt.exp_name else opt.ckpt, best_acc, path='%s/results.json' % (opt.save_dir))
    
    # for robustness experiments
    method = 'supcon' 
    if not os.path.isdir('./robustness/ckpt'):
        os.makedirs('./robustness/ckpt')
    torch.save(model.state_dict(), './robustness/ckpt/{}_encoder.pth'.format(method))
    torch.save(classifier.state_dict(), './robustness/ckpt/{}_classifier.pth'.format(method))

if __name__ == '__main__':
    main()
