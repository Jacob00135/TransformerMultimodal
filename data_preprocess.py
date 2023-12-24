import os
import pdb
import numpy as np
import pandas as pd
from collections import Counter, deque
from config import root_path


def compute_benefit(data):
    # 添加一列用于标识随访时间
    months = np.zeros(data.shape[0], 'int')
    for i, viscode in enumerate(data['VISCODE'].values):
        if viscode == 'bl':
            months[i] = 0
        else:
            months[i] = int(viscode[1:])
    data.insert(2, 'months', months)

    # 排序
    data = data.sort_values(['RID', 'months'])
    data.index = range(data.shape[0])

    # 计算benefit
    benefit = np.zeros(data.shape[0], 'float32')
    i = 0
    while i < data.shape[0]:
        rid = data.loc[i, 'RID']

        # 寻找第一个不是nan的ADAS13
        before_adas13 = data.loc[i, 'ADAS13']
        i = i + 1
        while pd.isna(before_adas13) and i < data.shape[0] and data.loc[i, 'RID'] == rid:
            benefit[i - 1] = np.nan
            before_adas13 = data.loc[i, 'ADAS13']
            i = i + 1

        # 没有找到不是nan的ADAS13
        if pd.isna(before_adas13):
            benefit[i - 1] = np.nan
            continue

        # 找到了第一个不是nan的ADAS13
        before_index = i - 1
        while i < data.shape[0] and data.loc[i, 'RID'] == rid:
            now_adas13 = data.loc[i, 'ADAS13']
            if pd.isna(now_adas13):
                benefit[i] = np.nan
            else:
                diff = before_adas13 - now_adas13
                if diff > 0:
                    benefit[before_index] = diff
                else:
                    benefit[before_index] = 0
                before_adas13 = now_adas13
                before_index = i
            i = i + 1
        benefit[before_index] = np.nan

    # 插入benefit字段
    data.insert(data.shape[1], 'benefit', benefit)
    data.index = range(data.shape[0])

    return data


def get_split_boolean(labels, split_ratio=(0.6, 0.2, 0.2)):
    if sum(split_ratio) != 1:
        raise 'sum(split_ratio)必须等于1'
    num_split = len(split_ratio)
    num_sample = len(labels)
    num_classes = len(set(labels))

    # 计算每个子集对于每一个类别的样本量
    category_num = Counter(labels)
    split_info = np.zeros((num_split, num_classes), dtype='int')
    for r in reversed(range(num_split)):
        for c in range(num_classes):
            if r == 0:
                split_info[r, c] = category_num[c] - sum(split_info[1:, c])
            else:
                split_info[r, c] = np.ceil(category_num[c] * split_ratio[r])

    # 分配数据到子集
    filter_boolean = np.zeros((num_split, num_sample), dtype='bool')
    distribute_queue = {}
    for c in range(num_classes):
        queue = []
        for r in reversed(range(num_split)):
            arr = np.zeros(split_info[r, c], dtype='int')
            arr[:] = r
            queue.extend(arr)
        distribute_queue[c] = deque(queue)
    for i, c in enumerate(labels):
        r = distribute_queue[c].popleft()
        filter_boolean[r, i] = True

    # 校验
    for r in range(num_split):
        sub_dataset_label = labels[filter_boolean[r, :]]
        sub_dataset_counter = Counter(sub_dataset_label)
        for c in range(num_classes):
            assert sub_dataset_counter[c] == split_info[r, c], '数据分割校验不通过'

    return filter_boolean


def split_dataset_consider_benefit(dataset, split_ratio=(0.6, 0.2, 0.2), shuffle=True):
    if sum(split_ratio) != 1:
        raise 'sum(split_ratio)必须等于1'

    # 打乱样本次序
    random_index = np.arange(dataset.shape[0], dtype='int')
    np.random.shuffle(random_index)
    dataset = dataset.iloc[random_index, :]
    dataset.index = range(dataset.shape[0])

    # 分割出训练集和测试集，保证测试集中的样本尽可能多地可计算benefit
    labels = dataset['DX'].values
    benefit = dataset['benefit'].values
    num_ad_benefit = sum((labels == 1) & (~np.isnan(benefit)))
    num_test_set_ad = int(np.ceil(sum(labels == 1) * split_ratio[2]))
    test_boolean = np.zeros(dataset.shape[0], dtype='bool')
    ad_benefit_index = np.where((labels == 1) & (~np.isnan(benefit)))[0]
    if num_ad_benefit >= num_test_set_ad:
        test_boolean[ad_benefit_index[:num_test_set_ad]] = True
    else:
        test_boolean[ad_benefit_index] = True
        require_num = num_test_set_ad - num_ad_benefit
        ad_non_benefit_index = np.where((labels == 1) & np.isnan(benefit))[0]
        test_boolean[ad_non_benefit_index[:require_num]] = True
    cn_index = np.where(labels == 0)[0]
    num_test_set_cn = int(np.ceil(sum(labels == 0) * split_ratio[2]))
    test_boolean[cn_index[:num_test_set_cn]] = True
    test_set = dataset[test_boolean]
    test_set.index = range(test_set.shape[0])
    train_set = dataset[~test_boolean]
    train_set.index = range(train_set.shape[0])

    # 把训练集再分割成训练集和验证集
    random_index = np.arange(train_set.shape[0], dtype='int')
    np.random.shuffle(random_index)
    train_set = train_set.iloc[random_index, :]
    train_set.index = range(train_set.shape[0])
    new_split_ratio = [0.0, 0.0]
    new_split_ratio[1] = split_ratio[1] / sum(split_ratio[:2])
    new_split_ratio[0] = 1.0 - new_split_ratio[1]
    filter_boolean = get_split_boolean(train_set['DX'].values, new_split_ratio)
    valid_set = train_set[filter_boolean[1, :]]
    valid_set.index = range(valid_set.shape[0])
    train_set = train_set[filter_boolean[0, :]]
    train_set.index = range(train_set.shape[0])

    return train_set, valid_set, test_set


def main():
    # 过滤原始数据的行和列
    src_data = pd.read_csv(os.path.join(root_path, 'datasets/multi_MRI_PET.csv'))
    src_data = src_data[['RID', 'VISCODE', 'DX', 'filename_MRI', 'filename_PET']].sort_values(by=['RID', 'VISCODE'])
    src_data = src_data[src_data['DX'] != 'MCI']

    # 获取benefit
    adni_merge = pd.read_csv(os.path.join(root_path, 'datasets/ADNIMERGE_without_bad_value.csv'))
    adni_merge = adni_merge[['RID', 'VISCODE', 'ADAS13']].sort_values(by=['RID', 'VISCODE'])
    adni_merge = compute_benefit(adni_merge)[['RID', 'VISCODE', 'benefit']].sort_values(by=['RID', 'VISCODE'])

    # 合并表
    dataset = pd.merge(src_data, adni_merge, how='left', on=['RID', 'VISCODE'])
    benefit = dataset['benefit'].values
    benefit[dataset['DX'] == 'CN'] = np.nan
    dataset['benefit'] = benefit
    dataset['DX'] = dataset['DX'].replace({'CN': 0, 'Dementia': 1})

    # 分割数据集并保存
    train_set, valid_set, test_set = split_dataset_consider_benefit(dataset, split_ratio=(0.6, 0.2, 0.2), shuffle=True)
    train_set.to_csv(os.path.join(root_path, 'datasets/train.csv'), index=False)
    valid_set.to_csv(os.path.join(root_path, 'datasets/valid.csv'), index=False)
    test_set.to_csv(os.path.join(root_path, 'datasets/test.csv'), index=False)


if __name__ == '__main__':
    main()
