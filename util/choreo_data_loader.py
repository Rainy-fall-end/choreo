from util.load_data import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

class choreo_dataset(Dataset):
    def __init__(self,file_path,back_len=24,forward_len=24) -> None:
        super().__init__()
        ds_all, ds_all_centered, datasets, datasets_centered, ds_counts = load_data(pattern=file_path)
        # datas = ds_all_centered[:,:,:3].reshape(ds_all_centered.shape[0],-1)
        dataset_0 = datasets_centered["betternot_and_retrograde"]
        datas = dataset_0.reshape(dataset_0.shape[0],-1)
        datas_res = datas[1:]-datas[:-1]
        mean_vals = np.mean(datas_res, axis=0)
        std_vals = np.std(datas_res, axis=0)
        # Z-Score 标准化
        normalized_data = (datas_res - mean_vals) / std_vals
        self.mean = mean_vals.reshape(53,3)
        self.std = std_vals.reshape(53,3)
        # datas_all = [datas[i:i+back_len+forward_len] for i in range(0,datas.shape[0],back_len+forward_len)]
        label = []
        tar = []
        for i in range(normalized_data.shape[0]-back_len-forward_len):
            label.append(normalized_data[i:i+back_len])
            tar.append(normalized_data[i+back_len:i+back_len+forward_len])
        self.label = np.stack(label)
        self.target = np.stack(tar)
        # datas_all.pop()
        # datas_all = np.stack(datas_all)
        
        # # datas_all = datas_all.reshape(datas_all.shape[0],-1)
        # self.label = datas_all[:,:back_len,:]
        # self.target =  datas_all[:,back_len:,:]
        self.label = torch.tensor(self.label, device="cuda",dtype = torch.float32)
        self.target = torch.tensor(self.target, device="cuda",dtype = torch.float32)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        return self.label[index],self.target[index]
    def get_ori(self,data):
        return data*self.std+self.mean
train_dataset = choreo_dataset(file_path="/root/autodl-tmp/transformer/choreo/data/mariel_betternot_and_retrograde.npy")
train_dataset_loader = DataLoader(train_dataset,batch_size=5,shuffle=False)
for label,target in train_dataset_loader:
    print(label.shape)
    print(target.shape)
    