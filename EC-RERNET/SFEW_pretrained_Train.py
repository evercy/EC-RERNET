'''Train RAF with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as transforms
import numpy as np
import os
import time
import argparse
import utils as utils
#from torchstat import stat
from data.SFEW import SFEW
from torch.autograd import Variable
from models import *

from thop import profile
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch RAF CNN Training')
parser.add_argument('--model', type=str, default='AntCNN', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='models/SFEW_nocrop', help='CNN architecture')
parser.add_argument('--train_bs', default=8, type=int, help='learning rate')
parser.add_argument('--test_bs', default=8, type=int, help='learning rate')
parser.add_argument('--lr', default=0.0004, type=float, help='learning rate')
parser.add_argument('--resume', default=False, type=int, help='resume from checkpoint')
parser.add_argument('--mixup', default=False, type=int, help='use mixup')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 50  # 50
learning_rate_decay_every = 30  # 5
learning_rate_decay_rate = 0.8  # 0.9

cut_size = 44
total_epoch = 300

total_prediction_fps = 0
total_prediction_n = 0

path = os.path.join(opt.dataset + '_' + opt.model)
writer = SummaryWriter(log_dir=os.path.join(opt.dataset + '_' + opt.model))

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.RandomCrop(44),
    transforms.Resize(44),
    transforms.RandomHorizontalFlip(),
    transforms.RandomChoice([
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, scale=(0.8, 1.2)),  # 在[0.8, 1.2]的范围内进行缩放
        transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率进行水平反射

        #transforms.RandomAffine(0, translate=(0.2, 0.2))  # 在[-2, 2]像素的范围内进行水平或垂直平移
    ]),
    # transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率进行水平反射
    # transforms.RandomAffine(0, scale=(0.8, 1.2)),  # 在[0.8, 1.2]的范围内进行缩放
    # transforms.RandomAffine(0, translate=(0.2, 0.2)) , # 在[-2, 2]像素的范围内进行水平或垂直平移
    transforms.ToTensor(),

    transforms.Normalize((0.33922732, 0.22370481, 0.17301577),
                         (0.1392488, 0.112497024, 0.101794325)),  # Augmentation
])

transform_test = transforms.Compose([
    #transforms.TenCrop(44),
    transforms.Resize(44),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.33871546, 0.22956325, 0.18020718], std=[0.13488474, 0.10898205, 0.10268902]),

])

trainset = SFEW(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_bs, shuffle=True, num_workers=1)

PrivateTestset = SFEW(split='PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.test_bs, shuffle=False, num_workers=1)


net = EC_RFERNet().cuda()
net = torch.load('/data2/chenyuan/cy/project/emo/CNN-RIS-master/models/RAF_no_power_AntCNN/PrivateTest_model_87125_8.pth')
print(net)

# 冻结模型的所有参数

for i, param in enumerate(net.parameters()):
    print(i)
    if i < 126:      # 前面一些参数冻结
        param.requires_grad = False



if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    Private_checkpoint = torch.load(os.path.join(path, 'PrivateTest_model.t7'))
    best_PrivateTest_acc = Private_checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = Private_checkpoint['best_PrivateTest_acc_epoch']

    print('best_PrivateTest_acc is ' + str(best_PrivateTest_acc))
    net.load_state_dict(Private_checkpoint['net'])
    start_epoch = Private_checkpoint['best_PrivateTest_acc_epoch'] + 1

else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)
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
        #utils.clip_gradient(optimizer, 0.1)
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
    t_prediction = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        t = time.time()
        test_bs, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        t_prediction += (time.time() - t)

        loss = criterion(outputs, targets)
        PrivateTest_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PrivateTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PrivateTest_loss / (batch_idx + 1), 100. * float(correct) / float(total), correct, total))
    total_prediction_fps = total_prediction_fps + (1 / (t_prediction / len(PrivateTestloader)))
    total_prediction_n = total_prediction_n + 1
    print('Prediction time: %.2f' % t_prediction + ', Average : %.5f/image' % (t_prediction / len(PrivateTestloader))
          + ', Speed : %.2fFPS' % (1 / (t_prediction / len(PrivateTestloader))))

    # Save checkpoint.
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
        #torch.save(net, os.path.join(path, 'PrivateTest_model'+{PrivateTest_acc:.2f}+'.pth'))
        torch.save(net, os.path.join(path, f'PrivateTest_model{PrivateTest_acc:.2f}.pth'))

        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

    return PrivateTest_loss / (batch_idx + 1), PrivateTest_acc


for epoch in range(start_epoch, total_epoch):
    train_loss, train_acc = train(epoch)
    valid_loss, valid_acc = PrivateTest(epoch)
    writer.add_scalars('epoch/loss', {'train': train_loss, 'valid': valid_loss}, epoch)
    writer.add_scalars('epoch/accuracy', {'train': train_acc, 'valid': valid_acc}, epoch)

print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)

print("total_prediction_fps: %0.2f" % total_prediction_fps)
print("total_prediction_n: %d" % total_prediction_n)
print('Average speed: %.2f FPS' % (total_prediction_fps / total_prediction_n))
writer.close()
