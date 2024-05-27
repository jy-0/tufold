import torch
import numpy as np
from load_datasets import get_one_sample_for_predict
from model import Model
import json
import os


model_config_path = '/your/path/here'
model_state_dict_path = '/your/path/here'

sample_path_list = [
    '/your/path/here/[short medium]/test',
]
results_path = '/your/path/here'


with open(model_config_path) as f:
    config_dict = json.load(f)['config']

num_embeddings = config_dict['num_embeddings']
padding_idx = config_dict['padding_idx']
d_model = config_dict['d_model']
d_k = config_dict['d_k']
d_v = config_dict['d_v']
h = config_dict['h']
d_ff = config_dict['d_ff']
N = config_dict['N']
seq_len = config_dict['seq_len']
disable_rcsa = config_dict['disable_rcsa']
use_low_high_resolution = config_dict['use_low_high_resolution']
low_high_resolution_patch_size = config_dict['low_high_resolution_patch_size']

device = 'mps'

model = Model(num_embeddings, padding_idx, d_model, d_k, d_v, h, d_ff, N, seq_len, disable_rcsa=disable_rcsa, 
              use_low_high_resolution=use_low_high_resolution, low_high_resolution_patch_size=low_high_resolution_patch_size).to(device)
model.load_state_dict(torch.load(model_state_dict_path, map_location=device), strict=False)
model.eval()

for sample_path in sample_path_list:
    print(f'Processing: {sample_path}')
    for i, file_name in enumerate(os.listdir(sample_path)):
        if i % 20 == 0:
            print(i)
        if file_name.startswith('.'):
            continue
        if file_name.endswith('_x.npy'):
            with torch.no_grad():
                x_npy = np.load(os.path.join(sample_path, file_name))
                x = get_one_sample_for_predict(x_npy, seq_len).to(device)
                pred = model(x)
                pred = pred.squeeze(0).cpu().numpy()
                main_name = file_name[:-6]
                np.save(os.path.join(results_path, main_name + '.npy'), pred)
