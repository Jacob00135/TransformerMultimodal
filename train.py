import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pdb
import torch
import numpy as np
import pandas as pd
from time import time as get_timestamp
from collections import Counter, deque
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, AddChanneld, ScaleIntensityd, EnsureTyped
from models import model_ad
from sklearn.metrics import roc_auc_score
from config import root_path, mri_path


def one_hot(array, num_classes=None):
    if num_classes is None:
        num_classes = len(set(array))
    num_sample = len(array)
    one_hot_array = np.zeros((num_sample, num_classes), dtype='int')
    for i, v in enumerate(array):
        one_hot_array[i, v] = 1
    return one_hot_array





def get_dataset_transform():
    train_transform = Compose([
        LoadImaged(keys=['MRI', 'PET']),
        AddChanneld(keys=['MRI', 'PET']),
        ScaleIntensityd(keys=['MRI', 'PET']),
        EnsureTyped(keys=['MRI', 'PET'])
    ])
    test_transform = Compose([
        LoadImaged(keys=['MRI', 'PET']),
        AddChanneld(keys=['MRI', 'PET']),
        ScaleIntensityd(keys=['MRI', 'PET']),
        EnsureTyped(keys=['MRI', 'PET'])
    ])
    return train_transform, test_transform


class ModelAD(object):

    def __init__(self, device, model_name):
        self.device = device
        self.model_name = model_name
        self.dataset_dir = os.path.join(root_path, 'datasets')
        self.checkpoint_save_dir = os.path.join(root_path, 'checkpoints', model_name)
        if not os.path.exists(self.checkpoint_save_dir):
            os.mkdir(self.checkpoint_save_dir)

        self.batch_size = 4
        self.num_classes = 2
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.num_train = 0
        self.num_valid = 0
        self.num_test = 0
        self.net = None
        self.optimizer = None
        self.loss = None
        self.epoch = 0
        self.steps = 0

    def dataset_dataframe_to_dict(self, dataset_dataframe):
        dataset_dataframe.index = range(dataset_dataframe.shape[0])
        dataset_dict = []
        for i in range(dataset_dataframe.shape[0]):
            dataset_dict.append({
                'MRI': os.path.join(mri_path, dataset_dataframe.loc[i, 'filename_MRI']),
                'PET': os.path.join(mri_path, dataset_dataframe.loc[i, 'filename_PET']),
                'label': dataset_dataframe.loc[i, 'DX']
            })

        return dataset_dict

    def dataset_dict_to_dataloader(self, dataset_dict, dataset_transform, shuffle=True):
        dataset = Dataset(data=dataset_dict, transform=dataset_transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)
        return dataloader

    def load_data(self, batch_size=4, shuffle=True):
        self.batch_size = batch_size
        train_transform, test_transform = get_dataset_transform()
        train_df = pd.read_csv(os.path.join(self.dataset_dir, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(self.dataset_dir, 'valid.csv'))
        test_df = pd.read_csv(os.path.join(self.dataset_dir, 'test.csv'))
        self.num_train = train_df.shape[0]
        self.num_valid = valid_df.shape[0]
        self.num_test = test_df.shape[0]
        self.train_dataloader = self.dataset_dict_to_dataloader(self.dataset_dataframe_to_dict(train_df), train_transform, shuffle=shuffle)
        self.valid_dataloader = self.dataset_dict_to_dataloader(self.dataset_dataframe_to_dict(valid_df), test_transform, shuffle=shuffle)
        self.test_dataloader = self.dataset_dict_to_dataloader(self.dataset_dataframe_to_dict(test_df), test_transform, shuffle=shuffle)

    def build_model(self, init_lr=0.0001):
        dim = 128
        self.net = model_ad(
            dim=dim,
            depth=3,
            heads=4,
            dim_head=dim // 4,
            mlp_dim=dim * 4,
            dropout=0
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=init_lr, weight_decay=0)
        self.loss = torch.nn.CrossEntropyLoss()

    def need_save_model(self):
        if self.epoch <= 10:
            return True
        if self.epoch <= 20 and self.steps % 5 == 0:
            return True
        if self.epoch <= 30 and self.steps % 10 == 0:
            return True
        if self.steps % 100 == 0:
            return True

    def save_model(self):
        torch.save(
            self.net.state_dict(),
            os.path.join(self.checkpoint_save_dir, 'model_ad_{}.pth'.format(self.steps))
        )

    def load_model(self, filename):
        model_path = os.path.join(self.checkpoint_save_dir, filename)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))

    def train_epoch(self):
        self.net.train()
        batch_start_time = get_timestamp()
        for batch_index, batch_data in enumerate(self.train_dataloader):
            batch_mri = batch_data['MRI'].to(self.device)
            batch_pet = batch_data['PET'].to(self.device)
            batch_label = batch_data['label'].to(self.device)

            self.optimizer.zero_grad()
            output_logits, mri_logits, pet_logits = self.net(batch_mri, batch_pet)
            ce_loss = self.loss(output_logits, batch_label)
            mri_gt = torch.ones([mri_logits.shape[0]], dtype=torch.int64).to(batch_mri.device)
            pet_gt = torch.zeros([pet_logits.shape[0]], dtype=torch.int64).to(batch_pet.device)
            ad_loss = (self.loss(mri_logits, mri_gt) + self.loss(pet_logits, pet_gt)) / 2
            all_loss = ad_loss + ce_loss
            all_loss.backward()
            self.optimizer.step()

            self.steps = self.steps + 1
            if self.need_save_model():
                self.save_model()

            # print('Epoch {} -- batch {} -- {:.2f}s'.format(self.epoch, batch_index + 1, get_timestamp() - batch_start_time))
            batch_start_time = get_timestamp()

            # TODO
            """
            if batch_index >= 1:
                break
            """

    def predict(self, stage):
        if stage == 'train':
            dataset = self.train_dataloader
            num_dataset = self.num_train
        elif stage == 'valid':
            dataset = self.valid_dataloader
            num_dataset = self.num_valid
        elif stage == 'test':
            dataset = self.test_dataloader
            num_dataset = self.num_test
        else:
            raise 'stage只能取：`train`，`valid`，`test`'

        self.net.eval()
        prediction = np.zeros((num_dataset, self.num_classes), dtype='float32')
        labels = np.zeros(num_dataset, dtype='int')
        batch_loss_list = []
        with torch.no_grad():
            for batch_index, batch_data in enumerate(dataset):
                batch_mri = batch_data['MRI'].to(self.device)
                batch_pet = batch_data['PET'].to(self.device)
                batch_label = batch_data['label'].to(self.device)
                output_logits, _, _ = self.net(batch_mri, batch_pet)
                batch_loss = self.loss(output_logits, batch_label)
                batch_loss_list.append(batch_loss.item())

                start = batch_index * self.batch_size
                end = start + self.batch_size
                prediction[start:end, :] = output_logits.cpu().squeeze().numpy()
                labels[start:end] = batch_label.cpu().squeeze().numpy()

                # TODO
                """
                if batch_index >= 1:
                    break
                """

        return prediction, labels, batch_loss_list

    def model_eval(self, stage):
        prediction, labels, batch_loss_list = self.predict(stage)
        auc = roc_auc_score(one_hot(labels), prediction)
        accuracy = sum(prediction.argmax(axis=1) == labels) / len(labels)
        loss = sum(batch_loss_list) / len(batch_loss_list)

        return auc, accuracy, loss

    def train(self, epochs=40):
        for epoch in range(1, epochs + 1):
            start_time = get_timestamp()

            self.epoch = epoch
            self.train_epoch()
            self.save_model()

            train_auc, train_accuracy, train_loss = self.model_eval(stage='train')
            valid_auc, valid_accuracy, valid_loss = self.model_eval(stage='valid')

            print('Epoch {} -- {:.2f}s'.format(self.epoch, get_timestamp() - start_time))
            print('train_loss={:.6f} -- valid_loss={:.6f}'.format(train_loss, valid_loss))
            print('train_auc={:.6f} -- valid_auc={:.6f}'.format(train_auc, valid_auc))
            print('train_accuracy={:.6f} -- valid_accuracy={:.6f}'.format(train_accuracy, valid_accuracy))

            # break  # TODO


if __name__ == '__main__':
    model = ModelAD(device=torch.device('cuda:1'), model_name='xjy_20231218')
    model.build_model()
    model.load_data(batch_size=4)
    model.train()
