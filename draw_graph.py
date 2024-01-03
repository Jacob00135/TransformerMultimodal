import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

sys.path.append(os.path.realpath('..'))
from config import root_path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DrawScatter(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw_full(self, figsize, dpi, xlabel, save_path, adjust_padding=None):
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.subplot(1, 1, 1)
        plt.scatter(
            self.x,
            self.y,
            marker='o', 
            c='#6868ff',
            s=15,
            lw=0.2,
            ec='#555555',
            zorder=2
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(np.arange(0.0, 1.1, 0.1))
        plt.yticks(np.arange(0.0, 1.1, 0.1))
        plt.xlabel(xlabel)
        plt.ylabel('benefit')
        plt.savefig(save_path)
        plt.grid(True, c='#eeeeee', ls='--', zorder=0)
        ax.set_aspect('equal', adjustable='box')
        if adjust_padding is not None:
            plt.subplots_adjust(**adjust_padding)

    def draw_magnify(self):
        pass

    def draw_subplot(self):
        pass


def draw_full_scatter(performance):
    config_map = {
        'accuracy': {
            'xlabel': '$Accuracy$'
        },
        'sensitivity': {
            'xlabel': '$Sensitivity$'
        },
        'specificity': {
            'xlabel': '$Specificity$'
        },
        'f1_score': {
            'xlabel': '$F1~Score$'
        },
        'auc': {
            'xlabel': '$AUC$'
        }
    }
    for variable_name, draw_config in config_map.items():
        df = performance[[variable_name, 'benefit']].sort_values(by=variable_name)
        x = df[variable_name].values
        y = df['benefit'].values
        ds = DrawScatter(x, y)
        ds.draw_full(
            figsize=(4, 4),
            dpi=1200,
            xlabel=draw_config['xlabel'],
            adjust_padding={'left': 0, 'bottom': 0, 'right': 0.99, 'top': 0.99},
            save_path=os.path.join(root_path, 'graph/full_{}.png'.format(variable_name))
        )


def draw_magnify_scatter(performance):
    pass


def draw_subplot_scatter(performance):
    pass


def get_performance(model_name):
    performance_path = os.path.join(root_path, 'datasets', model_name, 'performance.csv')
    performance = pd.read_csv(performance_path).sort_values(by='model_index')

    return performance


def main():
    graph_save_dir = os.path.join(root_path, 'graph')
    if not os.path.exists(graph_save_dir):
        os.mkdir(graph_save_dir)
    category_function_mapping = {
        'full': draw_full_scatter,
        'magnify': draw_magnify_scatter,
        'subplot': draw_subplot_scatter
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', help='绘图类型, 可选：full, magnify, subplot。默认：full')
    args = parser.parse_args()
    category = args.category
    if category not in category_function_mapping:
        category = 'full'
    function = category_function_mapping[category]
    performance = get_performance('xjy_20231218')
    function(performance)
    print('run: {}'.format(category))


if __name__ == '__main__':
    main()
