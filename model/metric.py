import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def f1_score(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        tp = 0
        fp = 0
        fn = 0
        for i in range(len(target)):
            if pred[i] == 1 and target[i] == 1:
                tp += 1
            elif pred[i] == 1 and target[i] == 0:
                fp += 1
            elif pred[i] == 0 and target[i] == 1:
                fn += 1
    return tp / (tp + 0.5 * (fp + fn))

def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        tp = 0
        fp = 0
        for i in range(len(target)):
            if pred[i] == 1 and target[i] == 1:
                tp += 1
            elif pred[i] == 1 and target[i] == 0:
                fp += 1
    return tp / (tp + fp)

def recall(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        tp = 0
        fn = 0
        for i in range(len(target)):
            if pred[i] == 1 and target[i] == 1:
                tp += 1
            elif pred[i] == 0 and target[i] == 1:
                fn += 1
    return tp / (tp + fn)