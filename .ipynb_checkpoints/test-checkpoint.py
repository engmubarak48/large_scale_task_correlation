
import torch
import numpy as np
from utils import scores, checkpoint
best_f1 = 0

def test_epoch(net, criterion, testloader, epoch, np_classes, outModelName):
    global best_f1
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    predicted_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.squeeze().cuda()

            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), targets.float())

            test_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            all_targets.append(targets.detach().cpu())
            predicted_labels.append(predicted.detach().cpu())
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx % 100 == 0:
                print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/(total * np_classes), correct, (total * np_classes)))

    # Save checkpoint.
    all_targets = np.concatenate(all_targets)
    predicted_labels = np.concatenate(predicted_labels)
    AP, f1 = scores(all_targets, predicted_labels)
    AP, f1 = 100.*AP, 100.*f1
    acc = 100.*correct/(total * np_classes)
    
    if f1 > best_f1:
        best_f1 = f1
        checkpoint(net, acc, epoch, outModelName)
    return (test_loss/batch_idx, acc, AP, f1)