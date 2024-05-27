import torch
from torch import nn
from torch.utils.data import DataLoader
from load_datasets import RNA10FSubsetsDataset
from model import Model
from optimization_loops import loop_train, loop_valid_test

import argparse
import json
import os

from time import time_ns

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--epoch_num', type=int, required=True)

    parser.add_argument('--num_embeddings', type=int, required=True)
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--h', type=int, required=True)
    parser.add_argument('--d_k', type=int, required=True)
    parser.add_argument('--d_v', type=int, required=True)
    parser.add_argument('--d_ff', type=int, required=True)
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--padding_idx', type=int, required=True)

    parser.add_argument('--dataset_root_path', type=str, required=True) 
    parser.add_argument('--sub_dataset_list', type=str, required=True, 
                        help="A string contains the names of sub-datasets splitted by ':'.")
    # short:medium

    parser.add_argument('--result_path', type=str, required=True, 
                        help='The path of a folder to save the results of experiment.')
    parser.add_argument('--model_and_log_main_name', type=str, required=True,
                        help='The main name of model state dict and log file. '
                             'Model name: main_name.pth, log name: main_name.json. '
                             'They will be saved to the folder suggested by --result_path.')

    # discard
    parser.add_argument('--disable_rcsa', action='store_true')
    # discard
    parser.add_argument('--use_patch', action='store_true')
    # discard
    parser.add_argument('--patch_size', type=int, default=5)
    # discard
    parser.add_argument('--use_gaussian_label_smoothing', action='store_true')
    # discard
    parser.add_argument('--use_low_high_resolution', action='store_true')
    # discard
    parser.add_argument('--low_high_resolution_patch_size', type=int, default=1)
    # discard
    parser.add_argument('--load_previous_model_state_dict', action='store_true')
    # discard
    parser.add_argument('--previous_model_state_dict_path', type=str, default='')

    # make sure to use this
    parser.add_argument('--train_use_class_indices_target', action='store_true')

    # discard
    parser.add_argument('--load_transformer_state_dict', action='store_true')
    # discard
    parser.add_argument('--transformer_state_dict_path', type=str, default='')
    # discard
    parser.add_argument('--load_cnn_state_dict', action='store_true')
    # discard
    parser.add_argument('--cnn_state_dict_path', type=str, default='')

    return parser.parse_args()

args = get_args()

batch_size = args.batch_size
lr = args.lr
epoch_num = args.epoch_num

num_embeddings = args.num_embeddings
d_model = args.d_model
h = args.h
d_k = args.d_k
d_v = args.d_v
d_ff = args.d_ff
N = args.N
seq_len = args.seq_len
padding_idx = args.padding_idx

dataset_root_path = args.dataset_root_path
sub_dataset_list = args.sub_dataset_list.split(':')

result_path = args.result_path
model_and_log_main_name = args.model_and_log_main_name

disable_rcsa = args.disable_rcsa

use_patch = args.use_patch
patch_size = args.patch_size

use_gaussian_label_smoothing = args.use_gaussian_label_smoothing

use_low_high_resolution = args.use_low_high_resolution
low_high_resolution_patch_size = args.low_high_resolution_patch_size

load_previous_model_state_dict = args.load_previous_model_state_dict
previous_model_state_dict_path = args.previous_model_state_dict_path

train_use_class_indices_target = args.train_use_class_indices_target

load_transformer_state_dict = args.load_transformer_state_dict
transformer_state_dict_path = args.transformer_state_dict_path
load_cnn_state_dict = args.load_cnn_state_dict
cnn_state_dict_path = args.cnn_state_dict_path

model_save_path = os.path.join(result_path, model_and_log_main_name + '.pth')
log_save_path = os.path.join(result_path, model_and_log_main_name + '.json')

train_dataset = RNA10FSubsetsDataset(dataset_root_path, sub_dataset_list, 'train', seq_len, 
                                     use_patch, patch_size,
                                     use_low_high_resolution, low_high_resolution_patch_size,
                                     train_use_class_indices_target)
valid_dataset = RNA10FSubsetsDataset(dataset_root_path, sub_dataset_list, 'valid', seq_len, 
                                     use_patch, patch_size,
                                     use_low_high_resolution, low_high_resolution_patch_size,
                                     train_use_class_indices_target)
test_dataset = RNA10FSubsetsDataset(dataset_root_path, sub_dataset_list, 'test', seq_len, 
                                    use_patch, patch_size,
                                    use_low_high_resolution, low_high_resolution_patch_size,
                                    train_use_class_indices_target)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')
print(f'Device: {device}')

model = Model(num_embeddings, padding_idx, d_model, d_k, d_v, h, d_ff, N, seq_len, disable_rcsa=disable_rcsa, 
              use_low_high_resolution=use_low_high_resolution, low_high_resolution_patch_size=low_high_resolution_patch_size).to(device)

# discard
if load_previous_model_state_dict:
    if os.path.exists(previous_model_state_dict_path):
        print(f'Load previous model from: {previous_model_state_dict_path}')
        model.load_state_dict(torch.load(previous_model_state_dict_path))
    else:
        print(f'Error: Path {previous_model_state_dict_path} does not exist. Exiting.')
        exit()

# discard
if load_transformer_state_dict:
    print(f'Load Transformer state dict: {transformer_state_dict_path}')
    model.transformer_encoder.load_state_dict(torch.load(transformer_state_dict_path))
if load_cnn_state_dict:
    print(f'Load CNN state dict: {cnn_state_dict_path}')
    model.conv_net.load_state_dict(torch.load(cnn_state_dict_path))


use_new_approach = train_use_class_indices_target
if use_new_approach:
    loss_fn = nn.CrossEntropyLoss(reduction='none')
else:
    loss_fn = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# logging
log_dict = {
    'config': {
        'batch_size': batch_size,
        'lr': lr,
        'epoch_num': epoch_num,
        'num_embeddings': num_embeddings,
        'd_model': d_model,
        'h': h,
        'd_k': d_k,
        'd_v': d_v,
        'd_ff': d_ff,
        'N': N,
        'seq_len': seq_len,
        'padding_idx': padding_idx,
        'dataset_root_path': dataset_root_path,
        'sub_dataset_list': sub_dataset_list,
        'result_path': result_path,
        'model_and_log_main_name': model_and_log_main_name,
        'disable_rcsa': disable_rcsa,
        'use_patch': use_patch,
        'patch_size': patch_size,
        'use_gaussian_label_smoothing': use_gaussian_label_smoothing,
        'use_low_high_resolution': use_low_high_resolution,
        'low_high_resolution_patch_size': low_high_resolution_patch_size,
        'load_previous_model_state_dict': load_previous_model_state_dict,
        'previous_model_state_dict_path': previous_model_state_dict_path,
        'train_use_class_indices_target': train_use_class_indices_target,
        'load_transformer_state_dict': load_transformer_state_dict,
        'transformer_state_dict_path': transformer_state_dict_path,
        'load_cnn_state_dict': load_cnn_state_dict,
        'cnn_state_dict_path': cnn_state_dict_path
    },
    'results': []
}


for epoch in range(epoch_num):
    print(f'Epoch: {epoch + 1}')

    # discard
    if use_gaussian_label_smoothing:
        if 0 <= epoch < 5:
            train_dataset.set_gaussian_label_smoothing(True, 5)
        elif 5 <= epoch < 10:
            train_dataset.set_gaussian_label_smoothing(True, 3)
        elif 10 <= epoch < 15:
            train_dataset.set_gaussian_label_smoothing(True, 2)
        elif 15 <= epoch < 20:
            train_dataset.set_gaussian_label_smoothing(True, 1.5)
        elif 20 <= epoch < 25:
            train_dataset.set_gaussian_label_smoothing(True, 1)
        elif 25 <= epoch < 30:
            train_dataset.set_gaussian_label_smoothing(True, 0.7)
        else:
            train_dataset.set_gaussian_label_smoothing(True, 0.3)

        print(f'Training dataset: sigma {train_dataset.sigma}')
    

    t0 = time_ns()
    log_dict_train = loop_train(model, train_loader, loss_fn, optimizer, use_new_approach=use_new_approach)
    t1 = time_ns()
    log_dict_valid = loop_valid_test(model, valid_loader)
    t2 = time_ns()
    log_dict_test = loop_valid_test(model, test_loader)
    t3 = time_ns()


    train_time = t1 - t0
    valid_time = t2 - t1
    test_time = t3 - t2

    print(f'Time: train {train_time * 1e-9}s, valid {valid_time * 1e-9}s, test {test_time * 1e-9}s')

    # logging
    log_dict['results'].append({
        'epoch': epoch,
        'train_log': log_dict_train,
        'valid_log': log_dict_valid,
        'test_log': log_dict_test,
        # record time
        'train_time': train_time,
        'valid_time': valid_time,
        'test_time': test_time
    })

    torch.save(model.state_dict(), model_save_path)

    # logging
    with open(log_save_path, 'w') as f:
        json.dump(log_dict, f, indent=2)

print('Done')
