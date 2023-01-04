

from sklearn.metrics import accuracy_score, f1_score, average_precision_score

def scores(y_true, y_pred):
    AP = average_precision_score(y_true, y_pred, average='samples')
    f1 = f1_score(y_true, y_pred, average='samples')
    return AP, f1


def checkpoint(model, acc, epoch, outModelName):
    # Save checkpoint.
    print('Saving..')
    state = {
        'state_dict': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpointPretrain'):
        os.mkdir('checkpointPretrain')
    torch.save(state, f'./checkpointPretrain/{outModelName}.t7')

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = base_learning_rate
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (base_learning_rate - 0.1) * epoch / 10.
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr