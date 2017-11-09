#! /usr/bin/env python
# encoding: utf-8
#这段程序是用来绘制ROC曲线以及计算AUC的
#宝宝加油，一定可以的！
#程序的输入是两个一维的数组
#scores和labels，分别代表的是分类器的得分，和数据的真实标签

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

figure_name = 1

def plot_roc(scores,labels):
    scores = np.array(scores)
    # Compute ROC curve and ROC area 
    y = np.array(labels)
    fpr,tpr,thresholds = metrics.roc_curve(y, scores)
    roc_auc = auc(fpr,tpr)
# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Plot of a ROC curve 
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    global figure_name
    plt.savefig('./figure/'+str(figure_name))
    figure_name=figure_name+1