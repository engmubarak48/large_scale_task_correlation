
import torch
import numpy as np
from utils import scores, checkpoint, 

def test(net, criterion, testloader, epoch, outModelName):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    predicted_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
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
                    % (test_loss/(batch_idx+1), 100.*correct/(total * 40), correct, (total * 40)))

    # Save checkpoint.
    all_targets = np.concatenate(all_targets)
    predicted_labels = np.concatenate(predicted_labels)
    AP, f1 = scores(all_targets, predicted_labels)

    acc = 100.*correct/(total * 40)
    if acc > best_acc:
        best_acc = acc
        checkpoint(net, acc, epoch, outModelName)
    return (test_loss/batch_idx, 100.*correct/(total * 40), 100.*AP, 100.*f1)