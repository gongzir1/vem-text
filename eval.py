from __future__ import print_function, absolute_import

__all__ = ['accuracy']

def accuracy_top1(output, target):
    """Computes the top-1 accuracy"""
    batch_size = target.size(0)

    # Get the index of the maximum predicted class (top-1)
    _, pred = output.topk(1, 1, True, True)  # Only take top-1 predictions
    pred = pred.t()  # Transpose the prediction tensor to match target shape

    # Compare the top-1 predictions with the true labels
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # Calculate the number of correct predictions and compute the accuracy
    correct_k = correct.reshape(-1).float().sum(0)  # Flatten and count correct predictions
    accuracy = correct_k.mul_(100.0 / batch_size)  # Compute accuracy in percentage

    return accuracy

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res