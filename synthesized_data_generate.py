# v25
import random
import os
import numpy as np
from collections import deque

def generate_synthesized_samples(min_seq_len: int, max_seq_len: int, sample_num: int, dataset_path: str):
    for i in range(sample_num):
        if i % 1000 == 0:
            print(f'Processing: {i}')
        rng = np.random.default_rng()
        this_seq_len = random.randint(min_seq_len, max_seq_len)
        sample_x = rng.integers(1, 5, size=this_seq_len)
        np.save(os.path.join(dataset_path, f'synthesized_{this_seq_len}_s_{i:010}_x.npy'), sample_x)

def synthesized_npy_to_fasta(npy_dataset_path: str, raw_dataset_path: str):
    print('Processing')
    for npy_sample_name in os.listdir(npy_dataset_path):
        if npy_sample_name.startswith('.'):
            continue
        npy_sample_main_name = os.path.splitext(npy_sample_name)[0][:-2]
        npy_sample = np.load(os.path.join(npy_dataset_path, npy_sample_name))
        str_sample = npy_sample_to_str(npy_sample)
        with open(os.path.join(raw_dataset_path, npy_sample_main_name + '.fasta'), 'w') as f:
            f.write(f'> {npy_sample_main_name}\n')
            f.write(str_sample + '\n')
    print('Done')

def linearfold_predict_to_npy(predict_path: str, npy_dataset_path: str):
    for predict_name in os.listdir(predict_path):
        if predict_name.startswith('.'):
            continue
        npy_main_name = os.path.splitext(predict_name)[0] + '_y.npy'
        with open(os.path.join(predict_path, predict_name)) as f:
            dot_parentheses_predict = f.readlines()[1].split()[0].strip()
        npy_predict_pairs = dot_parentheses_predict_to_pair_list(dot_parentheses_predict)
        np.save(os.path.join(npy_dataset_path, npy_main_name), npy_predict_pairs)

def npy_sample_to_str(npy_seq: np.ndarray) -> str:
    mapping_dict = {
        1: 'A',
        2: 'U',
        3: 'C',
        4: 'G'
    }
    return ''.join([mapping_dict[i] for i in npy_seq])

def dot_parentheses_predict_to_pair_list(predict: str) -> np.ndarray:
    idx_stack = deque()
    ch_stack = deque()
    pairing_list = []
    for idx, ch in enumerate(predict):
        if ch == '.':
            continue
        if ch == '(':
            idx_stack.append(idx)
            ch_stack.append(ch)
        elif ch == ')':
            ch_stack.pop()
            paired_idx = idx_stack.pop()
            pairing_list.append([paired_idx, idx])
            pairing_list.append([idx, paired_idx])
        else:
            raise RuntimeError(f'Invalid char {ch}.')
    assert len(idx_stack) == 0
    assert len(ch_stack) == 0
    return np.array(pairing_list)

if __name__ == '__main__':
    pass