from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


def true_positive(y_true, y_pred):
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    return tp


def true_negative(y_true, y_pred):
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    return tn


def false_positive(y_true, y_pred):
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    return fp


def false_negative(y_true, y_pred):
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    return fn


def macro_f1(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    f1 = 0
    for class_ in list(y_true.unique()):
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        tp = true_positive(temp_true, temp_pred)
        fn = false_negative(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)
        temp_recall = tp / (tp + fn + 1e-6)
        temp_precision = tp / (tp + fp + 1e-6)
        temp_f1 = 2 * temp_precision * temp_recall / (temp_precision + temp_recall + 1e-6)
        f1 += temp_f1
    f1 /= num_classes
    return f1


def micro_f1(y_true, y_pred):
    P = micro_precision_multiclass(y_true, y_pred)
    R = micro_recall_multiclass(y_true, y_pred)
    f1 = 2*P*R / (P + R)
    return f1


def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        other_class = [x for x in unique_class if x != per_class]
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)
        roc_auc_dict[per_class] = roc_auc
    return roc_auc_dict


# macro is better for balanced classes or when
# each class is equally important, regardless of the number of samples in each class
# micro is better for imbalanced datasets


def micro_recall_multiclass(y_true, y_pred):
    tp = 0
    fn = 0
    for class_ in y_true.unique():
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        tp += true_positive(temp_true, temp_pred)
        fn += false_negative(temp_true, temp_pred)
    recall = tp / (tp + fn)
    return recall


def macro_recall_multiclass(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    recall = 0
    for class_ in list(y_true.unique()):
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        tp = true_positive(temp_true, temp_pred)
        fn = false_negative(temp_true, temp_pred)
        temp_recall = tp / (tp + fn + 1e-6)
        recall += temp_recall
    recall /= num_classes
    return recall


def micro_precision_multiclass(y_true, y_pred):
    tp = 0
    fp = 0
    for class_ in y_true.unique():
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        tp += true_positive(temp_true, temp_pred)
        fp += false_positive(temp_true, temp_pred)
    precision = tp / (tp + fp)
    return precision


def macro_precision_multiclass(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    precision = 0
    for class_ in list(y_true.unique()):
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)
        temp_precision = tp / (tp + fp + 1e-6)
        precision += temp_precision
    precision /= num_classes
    return precision


def accuracy_multiclass(y_true, y_pred):
    correct = (y_pred == y_true).sum().item()
    return correct/len(y_true)


def resume_macro(y_true, y_pred):
    P = macro_precision_multiclass(y_true, y_pred)
    R = macro_recall_multiclass(y_true, y_pred)
    F = macro_f1(y_true, y_pred)
    A = accuracy_multiclass(y_true, y_pred)
    return P, R, F, A


def resume_micro(y_true, y_pred):
    P = micro_precision_multiclass(y_true, y_pred)
    R = micro_recall_multiclass(y_true, y_pred)
    F = micro_f1(y_true, y_pred)
    A = accuracy_multiclass(y_true, y_pred)
    return P, R, F, A


class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=-0.03, device='cpu'):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.device = device
        self.s = s
        self.m = m
        print("self.device_id", self.device_id)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):

        if self.device_id is None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.to(torch.device(f'{self.device}:0'))
            weight = sub_weights[0].to(torch.device(f'{self.device}:0'))
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.to(torch.device(f'{self.device}:{i}'))
                weight = sub_weights[i].to(torch.device(f'{self.device}:{i}'))
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x),
                                                     F.normalize(weight)).to(torch.device(f'{self.device}:0'))),
                                   dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size()).to(torch.device(self.device))
        if self.device_id is not None:
            one_hot = one_hot.to(torch.device(f'{self.device}:0'))
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot

        one_hot.scatter_(1, label.view(-1, 1).long(), 1).to()
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')
