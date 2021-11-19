import time



import torch
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def compute_acc(val_loader, device, model, criterion):
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    losses = AverageMeter()
    model.eval()
    if type(val_loader) == type([1]):
        dataloader = val_loader
        pass
    else:
        dataloader = [val_loader]

    with torch.no_grad():
        for tmp in dataloader:
            for i, (input, target) in enumerate(tmp):
                # measure data loading time

                input = input.to(device)
                target = target.to(device)

                # compute output
                output = model(input)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                val_top1.update(acc1.item(), input.size(0))
                val_top5.update(acc5.item(), input.size(0))

            # measure elapsed time

    return val_top1.avg, val_top5.avg
