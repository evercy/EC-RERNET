'''Train RAF with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms as transforms
import numpy as np
import os
import time
import argparse
import utils as utils
#from torchstat import stat
from torch.autograd import Variable
from models import *
import csv
from thop import profile
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch RAF CNN Training')
parser.add_argument('--model', type=str, default='EC-RFERNet', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='models/AffectNet', help='CNN architecture')
parser.add_argument('--train_bs', default=256, type=int, help='learning rate')
parser.add_argument('--test_bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', default=False, type=int, help='resume from checkpoint')
parser.add_argument('--mixup', default=False, type=int, help='use mixup')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# utils.setup_seed(0)
learning_rate_decay_start = 40  # 50
learning_rate_decay_every = 5  # 5
learning_rate_decay_rate = 0.75 # 0.9

cut_size = 44
total_epoch = 200

total_prediction_fps = 0
total_prediction_n = 0
from torch.utils.data import Dataset
import cv2
path = os.path.join(opt.dataset + '_' + opt.model)
writer = SummaryWriter(log_dir=os.path.join(opt.dataset + '_' + opt.model))
class CustomDataset(Dataset):
    def __init__(self, csv_file, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        #self.data = pd.read_csv(csv_file)
        self.data_label = []
        # 循环读取CSV中的数据
        #for index, row in self.data.iterrows():
        with open(csv_file, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                label = int(row['expression'])
                if label < 7:  # 仅保留 expression 不大于 7 的数据
                    fname = row['image']  # 将image转换为字符串
                    fname = fname + ".jpg"
                    image_path = os.path.join(base_dir, fname)
                    self.data_label.append((image_path, label))
                    #print('(image_path, label)', (image_path, label))
    def __len__(self):
        return len(self.data_label)

    def __getitem__(self, i):
        img_path, label = self.data_label[i]
        # 读取图像
        image = cv2.imread(img_path)
        #print(image.shape)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = int(label)
        return image, label

custom_dataset = CustomDataset(csv_file='/data1/liangle/lunwen/img/all_img/7.csv',
                               base_dir='/data1/liangle/lunwen/img/all_img/128_qx/128',
                               transform=None)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(44),
    transforms.ColorJitter(),
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率进行水平反射
        transforms.RandomAffine(0, scale=(0.8, 1.2)),  # 在[0.8, 1.2]的范围内进行缩放
        transforms.RandomRotation(30),
    ]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(44),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225]),
])

trainset = CustomDataset(csv_file='/data1/liangle/lunwen/img/all_img/7.csv',
                               base_dir='/data1/liangle/lunwen/img/all_img/128_qx/128',
                               transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_bs, shuffle=True, num_workers=8)

PrivateTestset = CustomDataset(csv_file='/data1/liangle/lunwen/img/test/test.csv',
                                    base_dir='/data1/liangle/lunwen/img/test/128/',
                                    transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.train_bs, shuffle=False, num_workers=8)

net = EC_RFERNet().cuda()

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
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.001)

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
