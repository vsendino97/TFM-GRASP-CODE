import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=2):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def conf_mat(output, target, conf_mat):
    #conf_mat = torch.zeros(output.shape[1], output.shape[1])
    pred = torch.argmax(output, dim=1)
    assert pred.shape[0] == len(target)
    for t, p in zip(target.view(-1), pred.view(-1)):
        conf_mat[t.long(), p.long()] += 1

def f1_score(precision, recall):
    f1 = torch.zeros([precision.shape[0]])
    for i in range(0, len(precision)):
        f1[i] = (2*precision[i]*recall[i])/ (precision[i]+recall[i])
    return torch.nan_to_num(f1)

