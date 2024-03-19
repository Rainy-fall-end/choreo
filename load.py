import torch
import argparse
import numpy as np
from models.model.transformer import Transformer
from util.plotting import animate_stick,load_data
from conf import *
from torch.utils.data import DataLoader
from util.choreo_data_loader import choreo_dataset
def read_res(data,res):
    data = data.reshape(1,data.shape[0],data.shape[1])
    reconstructed_data = data
    for diff in res:
        reconstructed_data = np.concatenate((reconstructed_data, (reconstructed_data[-1:] + diff[np.newaxis, :])), axis=0)
    return reconstructed_data
model = Transformer(src_pad_idx=-1,
                    trg_pad_idx=-1,
                    trg_sos_idx=-1,
                    d_model=d_model,
                    enc_voc_size=10000,
                    dec_voc_size=159,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)
model.load_state_dict(torch.load("model_file/24/best.pth"))
# model(input,target)
# input.shape = (training_length,1,feature_size)

ds_all, ds_all_centered, datasets, datasets_centered, ds_counts = load_data(pattern="/root/autodl-tmp/TSAT-main/Data/choreo/data/mariel_betternot_and_retrograde.npy")
ds_all_centered = ds_all_centered[:,:,:3]
train_dataset = choreo_dataset(file_path="/root/autodl-tmp/transformer/choreo/data/*.npy")
train_dataset_loader = DataLoader(train_dataset,batch_size=1,shuffle=False)
len = 24
for i in range(0,101,10):
    index = i
    batch = train_dataset[index-1]
    src = batch[0].unsqueeze(0)
    trg = batch[1].unsqueeze(0)
    output = model(src, trg[:, :-1])
    # output_reshape = output.contiguous().view(-1, output.shape[-1])
    trg = trg[:, 1:]
    output = output.cpu().detach().numpy().reshape(len-1,53,3)
    trg = trg.cpu().detach().numpy().reshape(len-1,53,3)
    output = train_dataset.get_ori(output)
    trg = train_dataset.get_ori(trg)
    animation = animate_stick(read_res(ds_all_centered[len+1+index,:,:3],trg), 
                            ghost = read_res(ds_all_centered[len+1+index,:,:3],output),
                            figsize=(10,8), 
                            cmap='inferno', 
                            cloud=False,
                            dot_size=10,
                            speed=1
                            )
    animation.save("res/"+"24_pre/res"+str(i)+".gif", writer='pillow')
