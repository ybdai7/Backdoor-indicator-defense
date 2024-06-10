import torch
import torch.utils.data
from torchvision import datasets, transforms

from dataloader.AbstractDataloader import AbstractDataloader
import random
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

import logging
import pickle

from utils.utils import NoiseDataset, RandomImages

logger = logging.getLogger("logger")

class WMFLDataloader(AbstractDataloader):
    def __init__(self, params):
        super(WMFLDataloader, self).__init__(params)
        self.load_dataset()
        self.create_loader()

    def load_dataset(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_ood = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_emnist = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor()
        ])

        if self.params["dataset"].upper() == "CIFAR10":
            self.train_dataset = datasets.CIFAR10("./data", train=True, download=True, 
                                                  transform=transform_train)
            self.test_dataset = datasets.CIFAR10("./data", train=False, download=True,
                                                 transform=transform_test)

            if self.params["ood_data_source"] == "CIFAR100":
                self.ood_dataset = datasets.CIFAR100("./data", train=True, download=True, 
                                                  transform=transform_ood)
            elif self.params["ood_data_source"] == "EMNIST":
                self.ood_dataset = datasets.EMNIST("./data", train=True, split="mnist", download=True,
                                            transform=transform_emnist)
            elif self.params["ood_data_source"] == "300KRANDOM":
                self.ood_dataset = RandomImages(transform=transform_ood, data_num=self.params["ood_data_sample_lens"])
            elif self.params["ood_data_source"] == "NOISE":
                self.ood_dataset = NoiseDataset(size=(3,32,32), num_samples=self.params["ood_data_sample_lens"])

        elif self.params["dataset"].upper() == "CIFAR100":
            self.train_dataset = datasets.CIFAR100("./data", train=True, download=True, 
                                                  transform=transform_train)
            self.test_dataset = datasets.CIFAR100("./data", train=False, download=True,
                                                 transform=transform_test)
            self.ood_dataset = datasets.CIFAR10("./data", train=True, download=True, 
                                                  transform=transform_ood)
        elif self.params["dataset"].upper() == "EMNIST":
            self.train_dataset = datasets.EMNIST("./data", train=True, split="mnist", download=True,
                                            transform=transform_emnist)
            self.test_dataset = datasets.EMNIST("./data", train=False, split="mnist", transform=transform_emnist)
            self.ood_dataset = datasets.CIFAR10("./data", train=True, download=True, 
                                                  transform=transform_ood)
        
        return True

    def _sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}

        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if self.params["semantic"] and (ind in self.params['poison_images'] or ind in self.params['poison_images_test']):
                continue
            
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list
    
    def _load_edge_case(self):
        with open('./data/edge-case/southwest_images_new_train.pkl', 'rb') as train_f:
            saved_southwest_dataset_train = pickle.load(train_f)
        with open('./data/edge-case/southwest_images_new_test.pkl', 'rb') as test_f:
            saved_southwest_dataset_test = pickle.load(test_f)        

        return saved_southwest_dataset_train, saved_southwest_dataset_test

    def _get_poison_train(self):
        indices = list()
        range_no_id = list(range(50000))
        for image in self.params['poison_images'] + self.params['poison_images_test']:
            if image in range_no_id and self.params['semantic']:
                range_no_id.remove(image)
        
        # add random images to other parts of the batch
        for batches in range(self.params["poison_no_reuse"]):
            range_iter = random.sample(range_no_id,
                                       self.params['poison_train_batch_size'])
            indices.extend(range_iter)

        ## poison dataset size 64 \times 200 (64: batch size, 200 batch)
        return torch.utils.data.DataLoader(self.train_dataset,
                           batch_size=self.params['poison_train_batch_size'],
                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                           drop_last=True)

    def _get_train(self, indices):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                    batch_size = self.params["train_batch_size"],
                                    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices))
                                    # drop_last=True)

    def _get_test(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                    batch_size = self.params["test_batch_size"],
                                    shuffle=True)

    def _get_global_dataloader(self):
        indices = list()
        for batches in range(self.params["global_no_reuse"]):
            if len(self.global_data_indices) == self.params["global_data_batch_size"]:
                range_iter = self.global_data_indices
            else:
                range_iter = random.sample(self.global_data_indices, self.params["global_data_batch_size"])
            # range_iter = self.global_data_indices
            indices.extend(range_iter)

        return torch.utils.data.DataLoader(self.train_dataset,
                                       batch_size=self.params["global_data_train_batch_size"],
                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                       drop_last=True)

    def _get_sample(self):
        r'''
        sample limited ood data as open set noise
        '''
        ood_data = list()
        ood_data_label = list()
        sample_index = random.sample(range(len(self.ood_dataset)), self.params["ood_data_sample_lens"])
        for ind in sample_index:
            ood_data.append(self.ood_dataset[ind])
            assigned_label = random.randint(0,9)
            ood_data_label.append(assigned_label)
        return ood_data, ood_data_label

    def _get_ood_dataloader(self):
        r'''
        sample limited ood data as open set noise
        '''
        indices = random.sample(range(len(self.ood_dataset)), self.params["ood_data_sample_lens"])

        ood_dataloader =  torch.utils.data.DataLoader(self.ood_dataset,
                                           batch_size=self.params["ood_data_batch_size"],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                           drop_last=True)
        ood_datalist = list(ood_dataloader)
        ood_datalist_shape = self.params["ood_data_sample_lens"]//self.params["ood_data_batch_size"] * self.params["ood_data_batch_size"] 
        assigned_labels = np.array([i for i in range(self.params["class_num"])] * \
            (ood_datalist_shape//self.params["class_num"]) + [i for i in range(ood_datalist_shape%self.params["class_num"])])
        np.random.shuffle(assigned_labels)
        assigned_labels = assigned_labels.reshape(self.params["ood_data_sample_lens"]//self.params["ood_data_batch_size"], self.params["ood_data_batch_size"])
        for batch_id, batch in enumerate(ood_datalist):
            data, targets = batch
            if self.params["dataset"].upper()=="EMNIST":
                ood_datalist[batch_id][0] = ood_datalist[batch_id][0][:,0,:,:].unsqueeze(axis=1)
            if self.params["ood_data_source"] == "EMNIST":
                ood_datalist[batch_id][0] = ood_datalist[batch_id][0].repeat(1,3,1,1)

            for ind in range(len(targets)):
                targets[ind] = assigned_labels[batch_id][ind]
        ood_dataloader=iter(ood_datalist)
        return ood_dataloader

    def create_loader(self):
        if self.params["sample_dirichlet"]:
            indices_per_participant_malicious = self._sample_dirichlet_train_data(
                    self.params["no_of_total_participants"],
                    alpha = 1000)

            indices_per_participant = self._sample_dirichlet_train_data(
                    self.params["no_of_total_participants"],
                    alpha = self.params["dirichlet_alpha"])

            for i in range(self.params["no_of_adversaries"]):
                indices_per_participant[i] = indices_per_participant_malicious[i]

            self.train_data = [self._get_train(indices) for pos, indices in
                                  indices_per_participant.items()]
            
        self.test_data = self._get_test()

        self.ood_data = self._get_ood_dataloader()
        self.poison_data = self._get_poison_train()

        self.edge_poison_train, self.edge_poison_test = self._load_edge_case()

        return True
