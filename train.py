import math
import time

from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from conf import *
from models.model.transformer import Transformer
from util.epoch_timer import epoch_time
from util.choreo_data_loader import choreo_dataset
import wandb
import os
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


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

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.SmoothL1Loss()


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch[0]
        trg = batch[1]

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        # output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:]

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def run(total_epoch, best_loss):
    train_dataset = choreo_dataset(file_path="choreo/data/mariel_betternot_and_retrograde.npy")
    train_dataset_loader = DataLoader(train_dataset,batch_size=36,shuffle=False)
    
    train_losses, test_losses, bleus = [], [], []
    # model.load_state_dict(torch.load("model_file/best_8.pth"))
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_dataset_loader, optimizer, criterion, clip)
        end_time = time.time()
        wandb.log({"losses": train_loss})
        train_losses.append(train_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.7f} | Train PPL: {math.exp(train_loss):7.3f}')
    torch.save(model.state_dict(),"model_file/24/best.pth")

if __name__ == '__main__':
    wandb.init(project="choreo")
    run(total_epoch=epoch, best_loss=inf)
    wandb.finish()