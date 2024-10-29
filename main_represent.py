from __future__ import print_function

import os
import sys
import argparse
import time
import math
import yaml
import random
import builtins
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

import uuid 
from losses import ConLoss
from utils.util import *
from utils.datasets import get_set

from networks.resnet_big import ConResNet
from networks.vgg_big import ConVGG
from networks.wrn_big import ConWRN
from networks.efficient_big import ConEfficientNet

import wandb


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--project_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./save/representation')
    parser.add_argument('--resume', help='path of model checkpoint to resume', type=str, 
                        default='')

    # dataset
    parser.add_argument('--dataset_name', type=str, default='flowers', choices=['cub', 'flowers', 'aircraft', 'car', 'cifar10', 'cifar100', 'tinyimagenet', 'imagenet', 'imagenet100'])
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
    parser.add_argument('--feat_dim', type=int, default=128, 
                        help='feature dimension for mlp')

    # optimization
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--precision', action='store_true', 
                        help='whether to use 16 bit precision or not')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # important arguments
    parser.add_argument('--method', type=str, 
                        choices=['Con', 'SupCon', 'SelfCon'], help='choose method')
    parser.add_argument('--multiview', action='store_true', 
                        help='use multiview batch or not')
    parser.add_argument('--label', action='store_false',
                        help='whether to use label information or not')
    parser.add_argument('--alpha', type=float, default=0.0, 
                        help='weight for selfcon with multiview loss function')

    # other arguments
    parser.add_argument('--randaug', action='store_true', 
                        help='whether to add randaugment or not')
    parser.add_argument('--weakaug', action='store_true', 
                        help='whether to use weak augmentation or not')

    opt = parser.parse_args()

    if opt.model.startswith('vgg'):
        if opt.selfcon_pos == '[False,False,False]':
            opt.selfcon_pos = '[False,False,False,False]'
        opt.selfcon_arch = 'vgg'
    elif opt.model.startswith('wrn'):
        if opt.selfcon_pos == '[False,False,False]':
            opt.selfcon_pos = '[False,False]'
        opt.selfcon_arch = 'wrn'
                
    # set the path according to the environment
    opt.model_path = '%s/%s/%s_models' % (opt.save_dir, opt.method, opt.dataset_name)

    if opt.dataset_name == 'cub':
       opt.n_cls = 200
    elif opt.dataset_name == 'cifar10':
        opt.n_cls = 10
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
        opt.cls = 102
    else:
        raise ValueError('dataset_name not supported: {}'.format(opt.dataset_name))
        
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.model_name = '{}_{}_{}_lr_{}_multiview_{}_label_{}_decay_{}_bsz_{}_temp_{}_seed_{}'.\
        format(opt.method, opt.dataset_name, opt.model, opt.learning_rate,
               opt.multiview, opt.label, opt.weight_decay, opt.batch_size, 
               opt.temp, opt.seed)

    # warm-up for large-batch training,
    if opt.batch_size >= 1024:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
            
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)            
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
    if opt.exp_name:
        opt.model_name = '{}_{}'.format(opt.model_name, opt.exp_name)
        
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def build_dataloader(opt):
    # construct data loader
    if opt.dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif opt.dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32
        
    elif opt.dataset_name in  ['cub', 'aircraft', 'car', 'flowers', 'cars']:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 224

    elif opt.dataset_name == 'tinyimagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 64
    elif opt.dataset_name == 'imagenet' or opt.dataset_name == 'imagenet100':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        size = 224
    else:
        raise ValueError('dataset_name not supported: {}'.format(opt.dataset_name))

    normalize = transforms.Normalize(mean=mean, std=std)
    transform = [transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)), transforms.RandomHorizontalFlip()] #transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)), 
    if not opt.weakaug:
        transform += [transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                      transforms.RandomGrayscale(p=0.2)]

    if not opt.dataset_name in ['cub', 'aircraft', 'flowers', 'cars']:
      transform += [normalize]
    else:
      transform += [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose(transform)

    if opt.randaug:
        train_transform.transforms.insert(0, RandAugment(2, 9))
    if opt.multiview:
        train_transform = TwoCropTransform(train_transform)
    
    train_dataset = get_set(opt, split='train', transform=train_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=None)

    return train_loader

def set_model(opt):
    model_kwargs = {'name': opt.model, 
                    'dataset': opt.dataset_name,
                    'selfcon_pos': eval(opt.selfcon_pos),
                    'selfcon_arch': opt.selfcon_arch,
                    'selfcon_size': opt.selfcon_size
                    }
    if opt.model.startswith('resnet'):
        model = ConResNet(**model_kwargs)
    elif opt.model.startswith('vgg'):
        model = ConVGG(**model_kwargs)
    elif opt.model.startswith('wrn'):
        model = ConWRN(**model_kwargs)
    elif opt.model.startswith('eff'):
        model = ConEfficientNet(**model_kwargs)
    print(model)
    criterion = ConLoss(temperature=opt.temp)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        
    return model, criterion, opt


def _train(images, labels, model, criterion, epoch, bsz, opt):
    # compute loss
    features = model(images)
    if opt.method == 'Con':
        f1, f2 = torch.split(features[1], [bsz, bsz], dim=0)
    elif opt.method == 'SupCon':
        if opt.multiview:
            f1, f2 = torch.split(features[1], [bsz, bsz], dim=0)
    else:   # opt.method == 'SelfCon'
        f1, f2 = features

    if opt.method == 'SupCon':
        # SupCon
        if opt.multiview:
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels)
        # SupCon-S
        else:
            features = features[1].unsqueeze(1)
            loss = criterion(features, labels, supcon_s=True)

    elif opt.method == 'Con':
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features)
    elif opt.method == 'SelfCon':
        loss = torch.tensor([0.0]).cuda()
        # SelfCon
        if not opt.multiview:
            if not opt.alpha:
                features = torch.cat([f.unsqueeze(1) for f in f1] + [f2.unsqueeze(1)], dim=1)
                # SelfCon-SU
                if not opt.label:
                    loss += criterion(features)
                # SelfCon
                else:
            
                    loss += criterion(features, labels)
            else:
                features = f2.unsqueeze(1)
                if opt.label:
                    loss += criterion(features, labels, supcon_s=True)

                features = torch.cat([f.unsqueeze(1) for f in f1] + [f2.unsqueeze(1)], dim=1)
                # SelfCon-SU*
                if not opt.label:
                    loss += opt.alpha * criterion(features, selfcon_s_FG=True) 
                # SelfCon-S*
                else:
                    loss += opt.alpha * criterion(features, labels, selfcon_s_FG=True)
        # SelfCon-M
        else: 
            if not opt.alpha:
                features = torch.cat([f.unsqueeze(1) for f in f1] + [f2.unsqueeze(1)], dim=1)
                labels_repeat = torch.cat([labels, labels], dim=0)
                # SelfCon-MU
                if not opt.label:
                    loss += criterion(features) 
                # SelfCon-M
                else:
                    loss += criterion(features, labels_repeat)
            else:
                f2_1, f2_2 = torch.split(f2, [bsz, bsz], dim=0)
                features = torch.cat([f2_1.unsqueeze(1), f2_2.unsqueeze(1)], dim=1)
                # contrastive loss between F (backbone)
                if not opt.label:
                    loss += criterion(features)
                else:
                    loss += criterion(features, labels)

                features = torch.cat([f.unsqueeze(1) for f in f1] + [f2.unsqueeze(1)], dim=1)
                # SelfCon-MU*
                if not opt.label:
                    loss += opt.alpha * criterion(features, selfcon_m_FG=True)
                # SelfCon-M* 
                else:
                    loss += opt.alpha * criterion(features, labels, selfcon_m_FG=True)
    else:
        raise ValueError('contrastive method not supported: {}'.
                         format(opt.method))
        
    return loss
    

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    if opt.precision:
        scaler = torch.cuda.amp.GradScaler()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = labels.shape[0]
        
        if opt.multiview:
            images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        if opt.precision:
            with torch.cuda.amp.autocast():
                loss = _train(images, labels, model, criterion, epoch, bsz, opt)
        else:
            loss = _train(images, labels, model, criterion, epoch, bsz, opt)
            
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        if not opt.precision:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg

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
    
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # build model and criterion
    model, criterion, opt = set_model(opt)
    
    # build data loader
    train_loader = build_dataloader(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)
    
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            opt.start_epoch += 1
            print("=> loaded checkpoint '{}' (epoch {})".format(opt.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    else:
        opt.start_epoch = 1
    
    start_time = time.time()
    # training routine
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        adjusted_lr = adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}, loss {:.3f}'.format(epoch, time2 - time1, loss))

        if opt.save_freq:
            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                        opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)

        # save the last model
        save_file = os.path.join(
            opt.save_folder, 'last.pth')
        save_model(model, optimizer, opt, epoch, save_file)

        wandb.log(
            {
                "loss": loss,
                "adjusted-lr": adjusted_lr
            }
        )
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
   
if __name__ == '__main__':
    main()
