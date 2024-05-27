import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
import numpy as np
from itertools import product


class RNA10FSubsetsDataset(Dataset):

    def __init__(self, root_folder_path: str, sub_datasets: list[str], dataset_type: str, max_seq_len: int,
                 use_patch=False, patch_size=5,
                 use_low_high_resolution=False, low_high_resolution_patch_size=1,
                 train_use_class_indices_target=False):

        self.max_seq_len = max_seq_len
        self.all_x_list = []
        self.all_y_list = []
        assert dataset_type in ('train', 'valid', 'test')

        # v8: we need dataset_type to provide different behaviors in __getitem__
        self.dataset_type = dataset_type

        for sub_dataset in sub_datasets:
            # print(f'{type(self).__name__} ({dataset_type}): Processing: {sub_dataset}')
            sample_root_path = os.path.join(root_folder_path, sub_dataset, dataset_type)

            main_name_set = set()
            for file_name in os.listdir(sample_root_path):
                if file_name.startswith('.'):
                    continue
                if file_name.endswith('.npy'):
                    main_name_set.add(file_name[:-6])
            for main_name in main_name_set:
                x_name = main_name + '_x.npy'
                y_name = main_name + '_y.npy'
                self.all_x_list.append(np.load(os.path.join(sample_root_path, x_name)))
                self.all_y_list.append(np.load(os.path.join(sample_root_path, y_name)))

        # discard
        self.train_use_gaussian_label_smoothing = False
        self.sigma = 0.0
        self.gaussian_values = None

        # discard
        self.use_patch = use_patch
        if self.use_patch:
            assert patch_size % 2 == 1

            self.patch_number_map = {}
            overall_idx_counter = 1
            for length in range(patch_size//2+1, patch_size+1):
                for p in product('01234', repeat=length):
                    original_patch = ''.join(p)
                    if length < patch_size:
                        self.patch_number_map['-'*(patch_size-length)+original_patch] = overall_idx_counter
                        overall_idx_counter += 1
                        self.patch_number_map[original_patch+'-'*(patch_size-length)] = overall_idx_counter
                        overall_idx_counter += 1
                    else:
                        self.patch_number_map[original_patch] = overall_idx_counter
                        overall_idx_counter += 1

        else:
            self.patch_number_map = None
        self.patch_size = patch_size

        # discard
        self.use_low_high_resolution = use_low_high_resolution
        self.low_high_resolution_patch_size = low_high_resolution_patch_size

        self.train_use_class_indices_target = train_use_class_indices_target

    # discard
    def set_gaussian_label_smoothing(self, train_use_gaussian_label_smoothing=False, sigma=0.1):
        if train_use_gaussian_label_smoothing:
            self.train_use_gaussian_label_smoothing = True
            self.sigma = sigma
            self.gaussian_values = torch.exp(-(torch.arange(-self.max_seq_len, self.max_seq_len+1)/self.sigma)**2)
        else:
            self.train_use_gaussian_label_smoothing = False
            self.sigma = 0.0
            self.gaussian_values = None

    def __len__(self):
        return len(self.all_x_list)
    
    def __getitem__(self, idx):
        
        x_npy = self.all_x_list[idx]
        y_npy = self.all_y_list[idx]


        if not self.use_patch:
            x = f.pad(torch.from_numpy(x_npy), pad=(0, self.max_seq_len - x_npy.size), value=5)
        else:
            x_list = []
            x_str = '-' * (self.patch_size // 2) + ''.join([str(i) for i in x_npy]) + '-' * (self.patch_size // 2)
            for i in range(x_npy.size):
                single_patch = x_str[i:i+self.patch_size]
                x_list.append(self.patch_number_map[single_patch])
            x = torch.tensor(x_list)
            x = f.pad(x, pad=(0, self.max_seq_len - x_npy.size), value=0)
        

        if self.dataset_type == 'train' and self.train_use_class_indices_target:

            y = np.arange(self.max_seq_len)
            if y_npy.size > 0:
                y[y_npy[:, 0]] = y_npy[:, 1]
            y = torch.from_numpy(y)
        else:
            y = torch.zeros((self.max_seq_len, self.max_seq_len))

            if y_npy.size > 0:

                if self.dataset_type == 'train' and self.train_use_gaussian_label_smoothing:
                    for row, col in y_npy:
                        start = self.max_seq_len - row
                        end = start + self.max_seq_len
                        y[:, col] = self.gaussian_values[start:end]
                else:
                    y[y_npy[:, 0], y_npy[:, 1]] = 1.0
            

            if self.use_low_high_resolution:
                y = f.max_pool2d(y.unsqueeze(0), self.low_high_resolution_patch_size).squeeze(0)

            if self.dataset_type == 'train':
                y[:, y.sum(dim=0) == 0] = 1 / y.size(0)

        return x, y

class RNA10FByRNAFamilyDataset(Dataset):

    def __init__(self, root_folder_path: str, sub_datasets: list[str], dataset_type: str, max_seq_len: int, rna_family_name_list: list[str]):

        assert dataset_type in ('train', 'valid', 'test')
        self.max_seq_len = max_seq_len
        self.all_x_list = []
        self.all_y_list = []
        self.dataset_type = dataset_type

        for sub_dataset in sub_datasets:
            sub_sub_dataset_root_path = os.path.join(root_folder_path, sub_dataset)

            for sub_sub_dataset in rna_family_name_list:
                sample_root_path = os.path.join(sub_sub_dataset_root_path, sub_sub_dataset)
                if not os.path.exists(sample_root_path):
                    continue
                
                main_name_set = set()

                for file_name in os.listdir(sample_root_path):
                    if file_name.startswith('.'):
                        continue
                    if file_name.endswith('.npy'):
                        main_name_set.add(file_name[:-6])
            
                for main_name in main_name_set:
                    x_name = main_name + '_x.npy'
                    y_name = main_name + '_y.npy'
                    self.all_x_list.append(np.load(os.path.join(sample_root_path, x_name)))
                    self.all_y_list.append(np.load(os.path.join(sample_root_path, y_name)))

    def __len__(self):
        return len(self.all_x_list)
    
    def __getitem__(self, idx):
        x_npy = self.all_x_list[idx]
        y_npy = self.all_y_list[idx]

        x = f.pad(torch.from_numpy(x_npy), pad=(0, self.max_seq_len - x_npy.size), value=5)

        if self.dataset_type == 'train':
            y = np.arange(self.max_seq_len)
            if y_npy.size > 0:
                y[y_npy[:, 0]] = y_npy[:, 1]
            y = torch.from_numpy(y)
        else:
            y = torch.zeros((self.max_seq_len, self.max_seq_len))
            if y_npy.size > 0:
                y[y_npy[:, 0], y_npy[:, 1]] = 1.0

        return x, y



class SynthesizedDataset(Dataset):

    def __init__(self, root_folder_path: str, dataset_type: str, max_seq_len: int):
        assert dataset_type in ('train', 'valid', 'test')
        self.max_seq_len = max_seq_len
        self.all_x_list = []
        self.all_y_list = []
        self.dataset_type = dataset_type

        main_name_set = set()

        for file_name in os.listdir(root_folder_path):
            if file_name.startswith('.'):
                continue
            if file_name.endswith('.npy'):
                main_name_set.add(file_name[:-6])
            
        for main_name in main_name_set:
            x_name = main_name + '_x.npy'
            y_name = main_name + '_y.npy'
            self.all_x_list.append(np.load(os.path.join(root_folder_path, x_name)))
            self.all_y_list.append(np.load(os.path.join(root_folder_path, y_name)))

    def __len__(self):
        return len(self.all_x_list)
    
    def __getitem__(self, idx):
        x_npy = self.all_x_list[idx]
        y_npy = self.all_y_list[idx]

        x = f.pad(torch.from_numpy(x_npy), pad=(0, self.max_seq_len - x_npy.size), value=5)

        if self.dataset_type == 'train':
            y = np.arange(self.max_seq_len)
            if y_npy.size > 0:
                y[y_npy[:, 0]] = y_npy[:, 1]
            y = torch.from_numpy(y)
        else:
            y = torch.zeros((self.max_seq_len, self.max_seq_len))
            if y_npy.size > 0:
                y[y_npy[:, 0], y_npy[:, 1]] = 1.0

        return x, y


class MergedDataset(Dataset):
    def __init__(self, dataset_list: list[Dataset], dataset_type: str, max_seq_len: int):
        assert dataset_type in ('train', 'valid', 'test')
        self.max_seq_len = max_seq_len
        self.all_x_list = []
        self.all_y_list = []
        self.dataset_type = dataset_type

        for dataset in dataset_list:
            self.all_x_list.extend(dataset.all_x_list)
            self.all_y_list.extend(dataset.all_y_list)
    
    def __len__(self):
        return len(self.all_x_list)
    
    def __getitem__(self, idx):
        x_npy = self.all_x_list[idx]
        y_npy = self.all_y_list[idx]

        x = f.pad(torch.from_numpy(x_npy), pad=(0, self.max_seq_len - x_npy.size), value=5)

        if self.dataset_type == 'train':
            y = np.arange(self.max_seq_len)
            if y_npy.size > 0:
                y[y_npy[:, 0]] = y_npy[:, 1]
            y = torch.from_numpy(y)
        else:
            y = torch.zeros((self.max_seq_len, self.max_seq_len))
            if y_npy.size > 0:
                y[y_npy[:, 0], y_npy[:, 1]] = 1.0

        return x, y


def get_one_sample_for_eval(x_npy, y_npy, max_seq_len):
    x = f.pad(torch.from_numpy(x_npy), pad=(0, max_seq_len - x_npy.size), value=5)
    y = torch.zeros((max_seq_len, max_seq_len))
    if y_npy.size > 0:
        y[y_npy[:, 0], y_npy[:, 1]] = 1.0
    return x.unsqueeze(0), y.unsqueeze(0)

def get_one_sample_for_predict(x_npy, max_seq_len):
    x = f.pad(torch.from_numpy(x_npy), pad=(0, max_seq_len - x_npy.size), value=5)
    return x.unsqueeze(0)


if __name__ == '__main__':
    pass
