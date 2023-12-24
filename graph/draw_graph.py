import os
import sys
import pdb
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.realpath('..'))
from config import root_path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    model_name = 'xjy_20231218'
    performance_path = os.path.join(root_path, 'datasets', model_name, 'performance.csv')
    performance = pd.read_csv(performance_path).sort_values(by='model_index')

    x = list(range(performance.shape[0]))
    y_name_list = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']
    for y_name in y_name_list:
        plt.figure(figsize=(4, 3), dpi=1200)
        plt.scatter(x, performance[y_name].values)
        plt.savefig(os.path.join(root_path, 'graph/static/images/{}.png'.format(y_name)))


if __name__ == '__main__':
    main()
