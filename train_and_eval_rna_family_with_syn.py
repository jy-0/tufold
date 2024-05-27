import torch
from torch import nn
from torch.utils.data import DataLoader
from load_datasets import RNA10FByRNAFamilyDataset, SynthesizedDataset, MergedDataset
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

    parser.add_argument('--result_path', type=str, required=True, 
                        help='The path of a folder to save the results of experiment.')

    parser.add_argument('--model_and_log_main_name', type=str, required=True,
                        help='The main name of model state dict and log file. '
                             'Model name: main_name.pth, log name: main_name.json. '
                             'They will be saved to the folder suggested by --result_path.')

    parser.add_argument('--rna_family_name_list_train', type=str, required=True,
                        help="A string contains the names of RNA family names splitted by ':'. "
                        "RNA family name should be one of '16srrna', '23srrna', '5srrna', 'introngpi', 'introngpii', "
                        "'rnasep', 'srp', 'telomerase', 'tmrna', 'trna'.")
    parser.add_argument('--rna_family_name_list_test', type=str, required=True,
                        help="A string contains the names of RNA family names splitted by ':'. "
                        "RNA family name should be one of '16srrna', '23srrna', '5srrna', 'introngpi', 'introngpii', "
                        "'rnasep', 'srp', 'telomerase', 'tmrna', 'trna'.")
    

    parser.add_argument('--synthesized_dataset_path', type=str, required=True)
    
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

rna_family_name_list_train = args.rna_family_name_list_train.split(':')
rna_family_name_list_test = args.rna_family_name_list_test.split(':')

synthesized_dataset_path = args.synthesized_dataset_path

model_save_path = os.path.join(result_path, model_and_log_main_name + '.pth')
log_save_path = os.path.join(result_path, model_and_log_main_name + '.json')


train_dataset = RNA10FByRNAFamilyDataset(dataset_root_path, sub_dataset_list, 'train', seq_len, rna_family_name_list_train)
synthesized_dataset = SynthesizedDataset(synthesized_dataset_path, 'train', seq_len)
final_train_dataset = MergedDataset([train_dataset, synthesized_dataset], 'train', seq_len)

test_dataset = RNA10FByRNAFamilyDataset(dataset_root_path, sub_dataset_list, 'test', seq_len, rna_family_name_list_test)

train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')
print(f'Device: {device}')

model = Model(num_embeddings, padding_idx, d_model, d_k, d_v, h, d_ff, N, seq_len).to(device)

loss_fn = nn.CrossEntropyLoss(reduction='none')
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
        'rna_family_name_list_train': rna_family_name_list_train,
        'rna_family_name_list_test': rna_family_name_list_test,
        'synthesized_dataset_path': synthesized_dataset_path
    },
    'results': []
}

for epoch in range(epoch_num):
    print(f'Epoch: {epoch + 1}')
    
    t0 = time_ns()
    log_dict_train = loop_train(model, train_loader, loss_fn, optimizer)
    t1 = time_ns()
    log_dict_test = loop_valid_test(model, test_loader)
    t2 = time_ns()

    train_time = t1 - t0
    test_time = t2 - t1

    print(f'Time: train {train_time * 1e-9}s, test {test_time * 1e-9}s')

    # logging
    log_dict['results'].append({
        'epoch': epoch,
        'train_log': log_dict_train,
        'test_log': log_dict_test,
        # record time
        'train_time': train_time,
        'test_time': test_time
    })

    torch.save(model.state_dict(), model_save_path)

    # logging
    with open(log_save_path, 'w') as f:
        json.dump(log_dict, f, indent=2)

print('Done')
