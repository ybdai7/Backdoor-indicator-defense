import torch
import numpy as np
import logging
import copy
import random
import datetime,os
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
logger = logging.getLogger("logger") 

def add_trigger(data, poisoned_pattern_choose=None, blend_pattern=None, blend_alpha=None,\
                evaluation=False, model_id=None):
    new_data = np.copy(data)
    channels, height, width = new_data.shape    

    if poisoned_pattern_choose == 1:
        for c in range(channels):
            new_data[c, 0, 2] = 2.5
            new_data[c, 1, 1] = 2.5
            new_data[c, 2, 0] = 2.5
            new_data[c, 2, 1] = 2.5
            new_data[c, 2, 2] = 2.5
    elif poisoned_pattern_choose == 10:
        new_data = (1-blend_alpha)*data + blend_alpha*blend_pattern
    elif poisoned_pattern_choose == 20:
        if evaluation:
            for c in range(channels):
                # for malicious 0
                new_data[c, 0, 1] = 2.5
                new_data[c, 0, 2] = 2.5
                new_data[c, 0, 3] = 2.5
                # for malicious 1
                new_data[c, 0, 6] = 2.5
                new_data[c, 0, 7] = 2.5
                new_data[c, 0, 8] = 2.5
                # for malicious 2
                new_data[c, 3, 1] = 2.5
                new_data[c, 3, 2] = 2.5
                new_data[c, 3, 3] = 2.5
                # for malicious 3
                new_data[c, 3, 6] = 2.5
                new_data[c, 3, 7] = 2.5
                new_data[c, 3, 8] = 2.5
        else:
            for c in range(channels):
                new_data[c, 0+3*(model_id//2), 1+5*(model_id%2)] = 2.5
                new_data[c, 0+3*(model_id//2), 2+5*(model_id%2)] = 2.5
                new_data[c, 0+3*(model_id//2), 3+5*(model_id%2)] = 2.5
    
    return torch.Tensor(new_data)

def save_model(name, folder_path, round, lr, ood_dataloader, save_on_round, model):
    logger.info(f"saving model: {name}")
    file_name = f"{folder_path}/saved_model_{name}_{round}.pt.tar"
    saved_dict = {
            "state_dict": model.state_dict(),
            "round": round,
            "lr": lr,
            "ood_dataloader": ood_dataloader
                  }
    if round in save_on_round:
        torch.save(saved_dict, file_name)

    return True

def plot_poisoned_acc(save_path, start_round, acc, acc_p, is_save_img=True):
    ind = [i+start_round for i in range(len(acc))] 
    plt.figure(dpi=300)
    plt.plot(ind, acc, label="main task acc")
    plt.plot(ind, acc_p, label="poisoned task acc")
    plt.legend()
    if is_save_img:
        plt.savefig(save_path+"/acc.png")
    
    return True

class NoiseDataset(torch.utils.data.Dataset):

    def __init__(self, size, num_samples):
        self.size = size
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        noise = torch.rand(self.size)
        noise = noise.cuda()
        return noise, 0

class RandomImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, data_num=50000):
        self.transform = transform
        self.data = np.load('./data/300K_random_images.npy').astype(np.uint8)

        if data_num != -1:
            all_id = list(range(len(self.data)))
            sample_id = random.sample(all_id, data_num)
            self.data = self.data[sample_id]

    def __getitem__(self, index):
        img = self.data[index]
        # img = np.transpose(img, (2,0,1))
        if self.transform is not None:
            img = self.transform(img)

        return img, 0 # 0 is the class

    def __len__(self):
        return len(self.data)