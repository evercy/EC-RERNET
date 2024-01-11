'''Train RAF with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import time
import argparse
import utils as utils
#from torchstat import stat
from data.fer import FER2013
from data.CK import CK
from torch.autograd import Variable
from models import *
from thop import profile
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='PyTorch RAF CNN Training')
parser.add_argument('--model', type=str, default='EC_RFERNet', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='models/CK', help='CNN architecture')
parser.add_argument('--train_bs', default=32, type=int, help='learning rate')
parser.add_argument('--test_bs', default=32, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', default=False, type=int, help='resume from checkpoint')
parser.add_argument('--mixup', default=False, type=int, help='use mixup')
opt = parser.parse_args()


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(44),
    transforms.Resize(44),
    #transforms.RandomHorizontalFlip(),
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率进行水平反射
        transforms.RandomAffine(0, scale=(0.8, 1.2)),  # 在[0.8, 1.2]的范围内进行缩放
        transforms.RandomRotation(5)
    ]),
    transforms.ToTensor(),
    transforms.Normalize((0.56010324, 0.43693307, 0.39122295),
                         (0.23726934, 0.21260591, 0.20737909)),  # Augmentation
])

transform_test = transforms.Compose([
    transforms.Resize(44),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.589499, 0.45687565, 0.40699387], std=[0.25357702, 0.23312956, 0.23275192]),

])

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        if opt.mixup:
            inputs, targets_a, targets_b, lam = utils.mixup_data(inputs, targets, 0.6, True)
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
        else:
            inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)

        if opt.mixup:
            loss = utils.mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)

        if opt.mixup:
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        else:
            correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss / (batch_idx + 1), 100. * float(correct) / float(total), correct, total))

    Train_acc = 100. * float(correct) / float(total)

    return train_loss / (batch_idx + 1), Train_acc


def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    global total_prediction_fps
    global total_prediction_n
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    #t_prediction = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        #t = time.time()
        test_bs, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        #outputs_avg = outputs.view(test_bs,  -1).mean(1)  # avg over crops
        _, predicted = torch.max(outputs.data, 1)
        #t_prediction += (time.time() - t)

        loss = criterion(outputs, targets)
        PrivateTest_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PrivateTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PrivateTest_loss / (batch_idx + 1), 100. * float(correct) / float(total), correct, total))

    total_prediction_n = total_prediction_n + 1

    PrivateTest_acc = 100. * float(correct) / float(total)
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'best_PrivateTest_acc': PrivateTest_acc,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(net, os.path.join(path, f'PrivateTest_model{i:.2f}.pth'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

    return PrivateTest_loss / (batch_idx + 1), PrivateTest_acc

use_cuda = torch.cuda.is_available()
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 30  # 50
learning_rate_decay_every = 10  # 5
learning_rate_decay_rate = 0.9  # 0.9

cut_size = 44
total_epoch = 80

total_prediction_fps = 0
total_prediction_n = 0

path = os.path.join(opt.dataset + '_' + opt.model)
writer = SummaryWriter(log_dir=os.path.join(opt.dataset + '_' + opt.model))

avg_val_acc = []
avg_test_epoch = []
for i in range(1,11):
    print('**' * 10, '第',i, '折', 'ing....', '**' * 10)
    trainset = CK(split='Training', fold=i, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_bs, shuffle=True, num_workers=1)

    PrivateTestset = CK(split='Testing', fold=i,transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.test_bs, shuffle=False,
                                                    num_workers=1)
    net = EC_RFERNet().cuda()
    #print(net)
    utils.setup_seed(0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    best_PrivateTest_acc = 0
    for epoch in range(start_epoch, total_epoch):
        train_loss, train_acc = train(epoch)
        valid_loss, valid_acc = PrivateTest(epoch)
        #print(train_acc)
        writer.add_scalars('epoch/loss', {'train': train_loss, 'valid': valid_loss}, epoch)
        writer.add_scalars('epoch/accuracy', {'train': train_acc, 'valid': valid_acc}, epoch)

    avg_val_acc.append(best_PrivateTest_acc)

    print('avg_val_acc',avg_val_acc[i-1])
    avg_test_epoch.append(best_PrivateTest_acc_epoch)
    print('best_PrivateTest_acc_epoch',avg_test_epoch[i-1])

print(avg_val_acc)
print('ALL_best_PrivateTest_acc_epoch',avg_test_epoch)
avg_val_acc = np.mean(avg_val_acc)

print("Average Validation Accuracy: {:.2f}%".format(avg_val_acc))

print("total_prediction_n: %d" % total_prediction_n)

writer.close()
