import pdb
import os
import torch
import numpy as np
import pandas as pd
from time import time as get_timestamp
from sklearn.metrics import roc_auc_score
from config import root_path, mri_path
from train import ModelAD, one_hot


class ConfusionMatrix(object):

    def __init__(self, y_true, y_pred):
        self.tp = 0
        self.fn = 0
        self.fp = 0
        self.tn = 0

        for tv, pv in zip(y_true, y_pred):
            if tv == 1:
                if tv == pv:
                    self.tp = self.tp + 1
                else:
                    self.fn = self.fn + 1
            elif tv != pv:
                self.fp = self.fp + 1
            else:
                self.tn = self.tn + 1

        self.cm = np.array([[self.tp, self.fn], [self.fp, self.tn]])

    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fn + self.fp + self.tn)

    def get_sensitivity(self):
        return self.tp / (self.tp + self.fn)

    def get_specificity(self):
        return self.tn / (self.tn + self.fp)

    def get_precision(self):
        return self.tp / (self.tp + self.fp)

    def get_recall(self):
        return self.tp / (self.tp + self.fn)

    def get_f1_score(self):
        precision = self.get_precision()
        recall = self.get_recall()
        return 2 * precision * recall / (precision + recall)


def main():
    # 初始化
    model_name = 'xjy_20231218'
    device = torch.device('cuda:1')
    checkpoint_save_dir = os.path.join(root_path, 'checkpoints', model_name)
    result = {
        'model_index': [],
        'model_name': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'f1_score': [],
        'auc': [],
        'benefit': []
    }
    start_time = get_timestamp()

    # 模型初始化
    model = ModelAD(device=device, model_name=model_name)
    model.build_model()
    model.load_data(batch_size=4, shuffle=False)
    benefit = pd.read_csv(os.path.join(root_path, 'datasets/test.csv'))['benefit'].fillna(0).values
    benefit = benefit / benefit.sum()

    checkpoint_filenames = os.listdir(checkpoint_save_dir)
    all_prediction = np.zeros((len(checkpoint_filenames), benefit.shape[0], 2), dtype='float32')
    for i, fn in enumerate(checkpoint_filenames):
        # 获取模型名称
        only_name, extend_name = fn.rsplit('.', 1)
        if extend_name != 'pth':
            continue
        model_index = int(only_name.rsplit('_', 1)[1])
        result['model_index'].append(model_index)
        result['model_name'].append(fn)

        # 载入模型
        model.load_model(os.path.join(checkpoint_save_dir, fn))

        # 预测并计算指标
        one_hot_prediction, labels, batch_loss_list = model.predict('test')
        all_prediction[i] = one_hot_prediction
        prediction = one_hot_prediction.argmax(axis=1)
        one_hot_labels = one_hot(labels)
        cm = ConfusionMatrix(labels, prediction)
        try:
            result['f1_score'].append(cm.get_f1_score())
        except ZeroDivisionError:
            result['model_index'].pop()
            result['model_name'].pop()
            print('忽略：model_name = {} -- {:.2f}s'.format(fn, get_timestamp() - start_time))
            start_time = get_timestamp()
            continue
        result['accuracy'].append(cm.get_accuracy())
        result['sensitivity'].append(cm.get_sensitivity())
        result['specificity'].append(cm.get_specificity())
        result['auc'].append(roc_auc_score(one_hot_labels, one_hot_prediction))
        result['benefit'].append(benefit[prediction == labels].sum())

        print('完成：model_name = {} -- {:.2f}s'.format(fn, get_timestamp() - start_time))
        start_time = get_timestamp()

    # 导出计算结果
    result = pd.DataFrame(result).sort_values(by='model_index')
    save_dir = os.path.join(root_path, 'datasets', model_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(os.path.join(save_dir, 'prediction.npy'), all_prediction)
    result.to_csv(os.path.join(save_dir, 'performance.csv'), index=False)


if __name__ == '__main__':
    main()
