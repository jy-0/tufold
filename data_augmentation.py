import numpy as np
import random
import os
import json
from copy import deepcopy
from collections import Counter

from typing import Callable, Tuple


def augment_by_rna_family_1(samples_path: str, results_path: str, rna_family_name: str):

    current_overall_idx_cnt = 0
    for cur_aug_sample_name in os.listdir(results_path):
        if cur_aug_sample_name.startswith('.'):
            continue
        if cur_aug_sample_name.endswith('_x.npy'):
            current_overall_idx_cnt += 1

    samples_to_be_augmented_main_name = []
    for sample_name in os.listdir(samples_path):
        if sample_name.startswith('.'):
            continue
        if sample_name.endswith('_x.npy'):
            sample_main_name = sample_name[:-6]
            if sample_main_name.startswith(rna_family_name):
                samples_to_be_augmented_main_name.append(sample_main_name)

    
    for ori_sample_main_name in samples_to_be_augmented_main_name:
        ori_x = np.load(os.path.join(samples_path, ori_sample_main_name + '_x.npy'))
        ori_y = np.load(os.path.join(samples_path, ori_sample_main_name + '_y.npy'))

        num_noise = int(ori_x.size * 0.1)
        no_base_pair_idx = list(set(range(ori_x.size)) - set(ori_y.reshape(-1)))
        idx_for_noise = random.sample(no_base_pair_idx, num_noise)
        aug_x = deepcopy(ori_x)
        aug_y = deepcopy(ori_y)
        for idx in idx_for_noise:
            aug_x[idx] = random.choice(list(set([1, 2, 3, 4]) - set([ori_x[idx]])))
        aug_main_name = f"{'aug1-' + rna_family_name}_{aug_x.size}_{'ori-' + ori_sample_main_name.split('_')[3]}_{current_overall_idx_cnt:010}"
        np.save(os.path.join(results_path, aug_main_name + '_x.npy'), aug_x)
        np.save(os.path.join(results_path, aug_main_name + '_y.npy'), aug_y)
        current_overall_idx_cnt += 1

def augment_by_rna_family_2(samples_path: str, results_path: str, rna_family_name: str,
                            min_aug_length=50):

    current_overall_idx_cnt = 0
    for cur_aug_sample_name in os.listdir(results_path):
        if cur_aug_sample_name.startswith('.'):
            continue
        if cur_aug_sample_name.endswith('_x.npy'):
            current_overall_idx_cnt += 1

    samples_to_be_augmented_main_name = []
    for sample_name in os.listdir(samples_path):
        if sample_name.startswith('.'):
            continue
        if sample_name.endswith('_x.npy'):
            sample_main_name = sample_name[:-6]
            if sample_main_name.startswith(rna_family_name):
                samples_to_be_augmented_main_name.append(sample_main_name)

    for ori_sample_main_name in samples_to_be_augmented_main_name:
        ori_x = np.load(os.path.join(samples_path, ori_sample_main_name + '_x.npy'))
        ori_y = np.load(os.path.join(samples_path, ori_sample_main_name + '_y.npy'))
        this_seq_max_remove_num = min(int(ori_x.size * 0.1), ori_x.size - min_aug_length)
        if this_seq_max_remove_num > 0:
            this_seq_remove_num = random.choice(range(1, this_seq_max_remove_num + 1))
        else:
            this_seq_remove_num = 0
        no_base_pair_idx = list(set(range(ori_x.size)) - set(ori_y.reshape(-1)))
        idx_for_remove = random.sample(no_base_pair_idx, this_seq_remove_num)
        idx_mapping_dict = {}
        for idx in range(ori_x.size):
            if idx in idx_for_remove:
                continue
            shift_before_cnt = 0
            for idx_rm in idx_for_remove:
                if idx_rm < idx:
                    shift_before_cnt += 1
            idx_mapping_dict[idx] = idx - shift_before_cnt
        
        removed_x_list = []
        for i, j in enumerate(ori_x):
            if i in idx_for_remove:
                continue
            removed_x_list.append(j)
        removed_x = np.array(removed_x_list)
        removed_y = deepcopy(ori_y)
        for i in range(removed_y.shape[0]):
            for j in range(removed_y.shape[1]):
                removed_y[i, j] = idx_mapping_dict[removed_y[i, j]]
        
        num_noise = int(removed_x.size * 0.1)
        no_base_pair_idx_2 = list(set(range(removed_x.size)) - set(removed_y.reshape(-1)))
        idx_for_noise = random.sample(no_base_pair_idx_2, num_noise)
        aug_x = deepcopy(removed_x)
        aug_y = deepcopy(removed_y)
        for idx in idx_for_noise:
            aug_x[idx] = random.choice(list(set([1, 2, 3, 4]) - set([removed_x[idx]])))
        aug_main_name = f"{'aug2-' + rna_family_name}_{aug_x.size}_{'ori-' + ori_sample_main_name.split('_')[3]}_{current_overall_idx_cnt:010}"
        np.save(os.path.join(results_path, aug_main_name + '_x.npy'), aug_x)
        np.save(os.path.join(results_path, aug_main_name + '_y.npy'), aug_y)
        current_overall_idx_cnt += 1


def augment_by_rna_family_3(samples_path: str, results_path: str, rna_family_name: str,
                            max_aug_length=500):
    current_overall_idx_cnt = 0
    for cur_aug_sample_name in os.listdir(results_path):
        if cur_aug_sample_name.startswith('.'):
            continue
        if cur_aug_sample_name.endswith('_x.npy'):
            current_overall_idx_cnt += 1

    samples_to_be_augmented_main_name = []
    for sample_name in os.listdir(samples_path):
        if sample_name.startswith('.'):
            continue
        if sample_name.endswith('_x.npy'):
            sample_main_name = sample_name[:-6]
            if sample_main_name.startswith(rna_family_name):
                samples_to_be_augmented_main_name.append(sample_main_name)

    for ori_sample_main_name in samples_to_be_augmented_main_name:
        ori_x = np.load(os.path.join(samples_path, ori_sample_main_name + '_x.npy'))
        ori_y = np.load(os.path.join(samples_path, ori_sample_main_name + '_y.npy'))
        assert max_aug_length >= ori_x.size
        this_seq_max_add_num = min(int(ori_x.size * 0.1), max_aug_length - ori_x.size)
        if this_seq_max_add_num > 0:
            this_seq_add_num = random.choice(range(1, this_seq_max_add_num + 1))
        else:
            this_seq_add_num = 0

        idx_for_add = random.choices(range(ori_x.size + 1), k=this_seq_add_num)
        idx_mapping_dict = {}
        for idx in range(ori_x.size):
            shift_before_cnt = 0
            for idx_add in idx_for_add:
                if idx_add <= idx:
                    shift_before_cnt += 1
            idx_mapping_dict[idx] = idx + shift_before_cnt
        
        added_x_list = []
        idx_for_add_counter = Counter(idx_for_add)
        for i, j in enumerate(ori_x):
            cur_idx_add_num = idx_for_add_counter[i]
            for _ in range(cur_idx_add_num):
                added_x_list.append(random.randint(1, 4))
            added_x_list.append(j)

        for _ in range(idx_for_add_counter[ori_x.size]):
            added_x_list.append(random.randint(1, 4))
        
        added_x = np.array(added_x_list)
        added_y = deepcopy(ori_y)
        for i in range(added_y.shape[0]):
            for j in range(added_y.shape[1]):
                added_y[i, j] = idx_mapping_dict[added_y[i, j]]
        
        num_noise = int(added_x.size * 0.1)
        no_base_pair_idx_2 = list(set(range(added_x.size)) - set(added_y.reshape(-1)))
        idx_for_noise = random.sample(no_base_pair_idx_2, num_noise)
        aug_x = deepcopy(added_x)
        aug_y = deepcopy(added_y)
        for idx in idx_for_noise:
            aug_x[idx] = random.choice(list(set([1, 2, 3, 4]) - set([added_x[idx]])))
        aug_main_name = f"{'aug3-' + rna_family_name}_{aug_x.size}_{'ori-' + ori_sample_main_name.split('_')[3]}_{current_overall_idx_cnt:010}"
        np.save(os.path.join(results_path, aug_main_name + '_x.npy'), aug_x)
        np.save(os.path.join(results_path, aug_main_name + '_y.npy'), aug_y)
        current_overall_idx_cnt += 1



def crop(x: np.ndarray, y: np.ndarray, 
         length_percentage=0.8, min_crop_length=10):

    x_length = x.size
    aug_x_length = int(x_length * length_percentage)
    if aug_x_length < min_crop_length:
        raise RuntimeError(
            f'Augmented sequence length {aug_x_length} is less than threshold {min_crop_length}.')

    start_idx = random.randint(0, x_length - aug_x_length)
    end_idx = start_idx + aug_x_length
    aug_x = x[start_idx:end_idx]
    aug_y = y[(y[:, 0] >= start_idx) & (y[:, 0] < end_idx) & (y[:, 1] >= start_idx) & (y[:, 1] < end_idx)]

    aug_y -= start_idx
    return aug_x, aug_y


def reverse(x: np.ndarray, y: np.ndarray):
    aug_x = np.flip(x)
    aug_y = x.size - 1 - y
    return aug_x, aug_y


def mask(x: np.ndarray, y: np.ndarray,
         mask_percentage=0.1, min_mask_number=1):

    pass


def augment_rna_family_dataset_concat(
        dataset_path: str,
        sub_folder_sub_sub_folder_name_list: list[list[str]],
        aug_sub_folder_name: str,
        aug_identifier: str,
        aug_sample_num: int,
        max_aug_seq_len=500,
        length_percentage_list=None,
        min_crop_length=10,
        log_folder_path=None):

    try:
        os.mkdir(os.path.join(dataset_path, aug_sub_folder_name))
    except FileExistsError:
        pass

    try:
        os.mkdir(os.path.join(dataset_path, aug_sub_folder_name, aug_identifier))
    except FileExistsError:
        pass

    sub_sub_folder_main_names_list = []
    for sub_folder_name, sub_sub_folder_name in sub_folder_sub_sub_folder_name_list:
        sub_sub_folder_path = os.path.join(dataset_path, sub_folder_name, sub_sub_folder_name)
        main_names_list = []
        for file_name in os.listdir(sub_sub_folder_path):
            if file_name.startswith('.'):
                continue
            if file_name.endswith('_x.npy'):
                main_names_list.append(file_name[:-6])
        sub_sub_folder_main_names_list.append(main_names_list)

    log_dict = {
        'overall_settings': {
                'dataset_path': dataset_path,
                'sub_folder_sub_sub_folder_name_list': sub_folder_sub_sub_folder_name_list
            }
        }
    
    for cnt in range(aug_sample_num):
        concat_main_names = [random.choice(l) for l in sub_sub_folder_main_names_list]
        concat_x_path_list = []
        concat_y_path_list = []
        for idx, main_name in enumerate(concat_main_names):
            concat_x_path_list.append(os.path.join(dataset_path,
                                                   sub_folder_sub_sub_folder_name_list[idx][0], 
                                                   sub_folder_sub_sub_folder_name_list[idx][1],
                                                   main_name + '_x.npy'))
            concat_y_path_list.append(os.path.join(dataset_path,
                                                   sub_folder_sub_sub_folder_name_list[idx][0], 
                                                   sub_folder_sub_sub_folder_name_list[idx][1],
                                                   main_name + '_y.npy'))
        
        concat_x_list = [np.load(path) for path in concat_x_path_list]
        concat_y_list = [np.load(path) for path in concat_y_path_list]
        
        concat_seq_len = sum(i.size for i in concat_x_list)
        len_ratio = 1
        if concat_seq_len > max_aug_seq_len:
            len_ratio = max_aug_seq_len / concat_seq_len
        if length_percentage_list is None:
            length_percentage_list = [1] * len(concat_x_list)
        final_length_percentage_list = [lp * len_ratio for lp in length_percentage_list]

        concat_cropped_x_list = []
        concat_cropped_y_list = []

        for idx, lp in enumerate(final_length_percentage_list):
            cropped_x, cropped_y = crop(concat_x_list[idx], concat_y_list[idx], 
                                        length_percentage=lp, min_crop_length=min_crop_length)
            concat_cropped_x_list.append(cropped_x)
            concat_cropped_y_list.append(cropped_y)
        
        aug_x = np.concatenate(concat_cropped_x_list, axis=0)

        current_base_for_y = 0
        for idx in range(len(concat_cropped_y_list)):
            concat_cropped_y_list[idx] += current_base_for_y
            current_base_for_y += concat_cropped_x_list[idx].size
        
        aug_y = np.concatenate(concat_cropped_y_list, axis=0)

        np.save(os.path.join(dataset_path, aug_sub_folder_name, aug_identifier, f'{aug_identifier}_{cnt:010}_x.npy'), aug_x)
        np.save(os.path.join(dataset_path, aug_sub_folder_name, aug_identifier, f'{aug_identifier}_{cnt:010}_y.npy'), aug_y)

        if log_folder_path is not None:
            log_dict[f'{aug_identifier}_{cnt:010}'] = {
                'concat_main_names': concat_main_names,
                'final_length_percentage_list': final_length_percentage_list
            }

    if log_folder_path is not None:
        with open(os.path.join(log_folder_path, f'{aug_identifier}_log.json'), 'w') as f:
            json.dump(log_dict, f, indent=2)


def augment_rna_family_dataset_crop(
        dataset_path: str,
        sub_folder_name: str,
        sub_sub_folder_name: str,
        aug_sub_folder_name: str,
        aug_identifier: str,
        aug_sample_num: int,
        max_aug_seq_len=500,
        length_percentage=0.8,
        min_crop_length=200,
        log_folder_path=None,
        max_try_times=None):
    try:
        os.mkdir(os.path.join(dataset_path, aug_sub_folder_name))
    except FileExistsError:
        pass

    try:
        os.mkdir(os.path.join(dataset_path, aug_sub_folder_name, aug_identifier))
    except FileExistsError:
        pass

    sub_sub_folder_path = os.path.join(dataset_path, sub_folder_name, sub_sub_folder_name)
    if max_try_times is None:
        max_try_times = 5 * aug_sample_num
    
    aug_sample_cnt = 0
    try_cnt = 0
    while True:
        pass



def augment_dataset(dataset_path: str, augment_func: Callable[..., Tuple[np.ndarray, np.ndarray]], 
                    identifier: str,
                    aug_sample_percentage=0.5, min_aug_sample_num=1,
                    **kwargs):

    for sub_dataset in os.listdir(dataset_path):
        if sub_dataset.startswith('.'):
            continue
        print(f'Processing: {sub_dataset}')
        sub_dataset_train_path = os.path.join(dataset_path, sub_dataset, 'train')
        main_name_list = []
        for sample_name in os.listdir(sub_dataset_train_path):
            if sample_name.startswith('.'):
                continue
            if sample_name.startswith('aug_'):
                # ignore existing augmented samples
                continue
            if sample_name.endswith('_x.npy'):
                main_name_list.append(sample_name[:-6])
        aug_sample_number = int(len(main_name_list) * aug_sample_percentage)
        if aug_sample_number < min_aug_sample_num:
            raise RuntimeError(f'Augmented sample number {aug_sample_number} '
                               f'is less than {min_aug_sample_num}.')
        print(f'Augmented sample number: {aug_sample_number}')
        main_name_to_be_augmented_list = random.sample(main_name_list, aug_sample_number)
        for main_name in main_name_to_be_augmented_list:
            original_x = np.load(os.path.join(sub_dataset_train_path, main_name + '_x.npy'))
            original_y = np.load(os.path.join(sub_dataset_train_path, main_name + '_y.npy'))
            aug_x, aug_y = augment_func(original_x, original_y, **kwargs)
            np.save(os.path.join(sub_dataset_train_path, f'aug_{identifier}_' + main_name + '_x.npy'), aug_x)
            np.save(os.path.join(sub_dataset_train_path, f'aug_{identifier}_' + main_name + '_y.npy'), aug_y)
    print('Done.')


def remove_all_augmented_samples(dataset_path, verbose=True):

    for sub_dataset in os.listdir(dataset_path):
        if sub_dataset.startswith('.'):
            continue
        print(f'Processing: {sub_dataset}')
        sub_dataset_train_path = os.path.join(dataset_path, sub_dataset, 'train')
        for sample_name in os.listdir(sub_dataset_train_path):
            if sample_name.startswith('aug_'):
                sample_to_remove_path = os.path.join(sub_dataset_train_path, sample_name)
                if verbose:
                    print(f'Removing: {sample_to_remove_path}')
                os.remove(sample_to_remove_path)


if __name__ == '__main__':
    pass