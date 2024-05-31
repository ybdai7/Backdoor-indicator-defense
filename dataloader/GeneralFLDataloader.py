import torch
import torch.utils.data
from torchvision import datasets, transforms

from dataloader.AbstractDataloader import AbstractDataloader
import random
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

class GeneralFLDataloader(AbstractDataloader):
    def __init__(self, params):
        super(GeneralFLDataloader, self).__init__(params)
        self.load_dataset()
        self.create_loader()

    def load_dataset(self):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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
        elif self.params["dataset"].upper() == "CIFAR100":
            self.train_dataset = datasets.CIFAR100("./data", train=True, download=True, 
                                                  transform=transform_train)
            self.test_dataset = datasets.CIFAR100("./data", train=False, download=True,
                                                 transform=transform_test)
        elif self.params["dataset"].upper() == "EMNIST":
            self.train_dataset = datasets.EMNIST("./data", train=True, split="mnist", download=True,
                                            transform=transform_emnist)
            self.test_dataset = datasets.EMNIST("./data", train=False, split="mnist", transform=transform_emnist)
        
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

    def _get_train(self, indices):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                    batch_size = self.params["train_batch_size"],
                                    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices))

    def _get_test(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                    batch_size = self.params["test_batch_size"],
                                    shuffle=True)

    def create_loader(self):
        if self.params["sample_dirichlet"]:
            indices_per_participant = self._sample_dirichlet_train_data(
                    self.params["no_of_total_participants"],
                    alpha = self.params["dirichlet_alpha"])

            self.train_data = [self._get_train(indices) for pos, indices in
                                  indices_per_participant.items()]
            
        self.test_data = self._get_test()

        return True
