

import torch
import numpy as np
from utils import scores
from torch.autograd import Variable

def train_epoch(net, criterion, optimizer, trainloader, np_classes, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    predicted_labels = []
    all_targets = []
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
    
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets.squeeze())
        outputs = net(inputs)
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = torch.round(torch.sigmoid(outputs))
        
        all_targets.append(targets.detach().cpu())
        predicted_labels.append(predicted.detach().cpu())
        
        total += targets.size(0)
        assert predicted.shape == targets.shape
        correct += predicted.eq(targets.data).cpu().sum()
        
        if batch_idx % 100 == 0:
            print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/(total * np_classes), correct, (total * np_classes)))
    
    all_targets = np.concatenate(all_targets)
    predicted_labels = np.concatenate(predicted_labels)
    AP, f1 = scores(all_targets, predicted_labels)
    AP, f1 = 100.*AP, 100.*f1
    acc = 100.*correct/(total * np_classes)
    return (train_loss/batch_idx, acc, AP, f1)

