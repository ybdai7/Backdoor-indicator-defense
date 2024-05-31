import torch
import torch.nn as nn
from torchvision import transforms
from participants.clients.AbstractClient import AbstractClient

import numpy as np
import random
import logging
import time
import math
import copy
logger = logging.getLogger("logger")

import models.vgg
import models.resnet

from utils.utils import add_trigger
from utils.losses import SupConLoss

class ChameleonMaliciousClient(AbstractClient):
    def __init__(self, params, train_dataset, blend_pattern, open_set, 
                 edge_case_train, edge_case_test, open_set_label=None):
        super(ChameleonMaliciousClient, self).__init__(params)
        self.train_dataset = train_dataset
        sample_data, _ = self.train_dataset[1]
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test

        self.blend_pattern = blend_pattern
        self.open_set = open_set
        self.open_set_label = open_set_label
        self._create_contrastive_model()
        self._loss_function()

    def _create_contrastive_model(self):
        r"""
        create global model according to the uploaded params info,
        ATTENTION: VGG model does not support EMNIST task YET!
        """
        if "ResNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                contrastive_model = getattr(models.resnet, f"SupCon{self.params['model_type']}")(dataset="CIFAR")
            elif self.params["dataset"].upper() == "CIFAR100":
                contrastive_model = getattr(models.resnet, f"SupCon{self.params['model_type']}")(dataset="CIFAR")
            elif self.params["dataset"].upper() == "EMNIST":
                contrastive_model = getattr(models.resnet, f"SupCon{self.params['model_type']}")(dataset="EMNIST")
        elif "VGG" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                contrastive_model = getattr(models.vgg, f"SupCon{self.params['model_type']}")
            elif self.params["dataset"].upper() == "CIFAR100":
                contrastive_model = getattr(models.vgg, f"SupCon{self.params['model_type']}")
        
        self.contrastive_model = contrastive_model.cuda()
        return True

    def _loss_function(self):
        self.ce_loss = nn.functional.cross_entropy
        self.ceriterion = nn.functional.cross_entropy
        self.supcon_loss = SupConLoss().cuda()
        return True

    def _ce_optimizer(self):
        self.ce_optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, self.local_model.parameters()), lr=self.params["poisoned_lr"],
                                    momentum=self.params["poisoned_momentum"], weight_decay=self.params["poisoned_weight_decay"])  
        return True

    def _supcon_optimizer(self): 
        self.supcon_optimizer = torch.optim.SGD(self.contrastive_model.parameters(), lr=self.params["poisoned_supcon_lr"],
                                    momentum=self.params["poisoned_supcon_momentum"], weight_decay=self.params["poisoned_supcon_weight_decay"])  
        return True

    def _ce_scheduler(self):
        self.ce_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.ce_optimizer,
                                                 milestones=self.params['malicious_milestones'],
                                                 gamma=self.params['malicious_lr_gamma'])
        return True

    def _supcon_scheduler(self):
        self.supcon_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.supcon_optimizer,
                                                 milestones=self.params['malicious_supcon_milestones'],
                                                 gamma=self.params['malicious_supcon_lr_gamma'])
        return True

    def _model_dist_norm(self, model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    def _model_dist_norm_var(self, model, target_params_variables, norm=2):
        size = 0
        for name, layer in model.named_parameters():
            size += layer.view(-1).shape[0]
        sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            sum_var[size:size + layer.view(-1).shape[0]] = (
            layer - target_params_variables[name]).view(-1)
            size += layer.view(-1).shape[0]

        return torch.norm(sum_var, norm)

    def batch_label_distrib(self, targets):
        distrib_dict=dict()
        no_class = 100 if self.params["dataset"].upper()=="CIFAR100" else 10
        for label in range(no_class):
            distrib_dict[label] = 0
        sum_no = 0
        
        for label in targets:
            label = label.item()
            distrib_dict[label] += 1
            sum_no+=1

        percentage_dict=dict()
        for key,value in distrib_dict.items():
            percentage_dict[key] = round(value/sum_no, 2)

        return distrib_dict, percentage_dict, sum_no

    def _projection(self, target_params_variables, model):

        model_norm = self._model_dist_norm(model, target_params_variables)
        if self.params["show_train_log"]:
            logger.info(f"model dist is :{model_norm}")

        if model_norm > self.params["poisoned_projection_norm"] and self.params["poisoned_is_projection_grad"]:
            norm_scale = self.params["poisoned_projection_norm"] / model_norm
            for name, param in model.named_parameters():
                clipped_difference = norm_scale * (
                        param.data - target_params_variables[name])
                param.data.copy_(target_params_variables[name]+clipped_difference)

        return True

    def local_training(self, train_data, test_data, target_params_variables, global_data, is_log_train, poisoned_pattern_choose=None, round=None, model_id=None):
        data_iterator = train_data
        self._loss_function()

        self.contrastive_model.copy_params(self.local_model.state_dict())
        self._supcon_optimizer()
        self._supcon_scheduler()

        for internal_round in range(self.params["poisoned_supcon_retrain_no_times"]):
            for batch_id, batch in enumerate(data_iterator):
                self.supcon_optimizer.zero_grad()
                batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=False, model_id=model_id)
                data, targets = batch
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                output = self.contrastive_model(data)
                contrastive_loss = self.supcon_loss(output, targets,
                                                    scale_weight=self.params["fac_scale_weight"],
                                                    fac_label=self.params["poison_label_swap"])
                distance_loss = self._model_dist_norm_var(self.contrastive_model, target_params_variables)
                loss = contrastive_loss + (self.params["Fedprox_mu"]/2) * distance_loss
                loss.backward()
                self.supcon_optimizer.step()
                self._projection(target_params_variables, model=self.contrastive_model)

            self.supcon_scheduler.step()

        self.local_model.copy_params(self.contrastive_model.state_dict())
        for params in self.local_model.named_parameters():
            if params[0] != "linear.weight" and params[0] != "linear.bias":
                params[1].require_grad = False

        self._ce_optimizer()
        self._ce_scheduler()

        for internal_round in range(self.params["poisoned_retrain_no_times"]):
            for batch_id, batch in enumerate(data_iterator):
                self.ce_optimizer.zero_grad()
                batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=False)
                data, targets = batch
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                output = self.local_model(data)
                class_loss = self.ce_loss(output, targets)
                distance_loss = self._model_dist_norm_var(self.local_model, target_params_variables)
                loss = class_loss
                loss.backward()
                self.ce_optimizer.step()
                self._projection(target_params_variables, model=self.local_model)

            self.ce_scheduler.step()

        for params in self.local_model.named_parameters():
            params[1].requires_grad = True

    def _watermarking_batch_injection(self, batch, evaluation=False):
        r'''
        poisoned_pattern_choose:
        evaluation:
        Open-set data: CIFAR10 for CIFAR100; CIFAR100 for CIFAR10
        noise_label_pattern: 0 for close-set; 1 for open-set
        '''
        poisoned_batch = copy.deepcopy(batch)
        batch_length = len(batch[0])
        poisoned_len = int(batch_length*self.params["noise_rate"]) if not evaluation else batch_length

        if self.params['noise_pattern']==0:
            logger.info(f"open_set is None")
            for pos in range(poisoned_len):
                rand=random.randint(0,self.params["class_num"]-1)
                while rand==poisoned_batch[1][pos]:rand=random.randint(0,self.params["class_num"]-1)
                poisoned_batch[1][pos]=random.randint(0,self.params["class_num"]-1)
        elif self.params['noise_pattern']==1:
            index=random.sample(range(len(self.open_set)),poisoned_len)
            for pos in range(poisoned_len):
                poisoned_batch[0][pos]=self.open_set[index[pos]][0]
                if self.params["noise_label_fixed"]:
                    # poisoned_batch[1][pos]=self.open_set[index[pos]][1]//10
                    poisoned_batch[1][pos]=self.open_set_label[index[pos]]
                else:
                    poisoned_batch[1][pos]=random.randint(0,self.params["class_num"]-1)

        return poisoned_batch, batch

    def _poisoned_batch_injection(self, batch, poisoned_pattern_choose=None, evaluation=False, model_id=None):
        r"""
        replace the poisoned batch with the oirginal batch
        """
        poisoned_batch = copy.deepcopy(batch)
        original_batch = copy.deepcopy(batch)
        poisoned_len = self.params["poisoned_len"] if not evaluation else len(poisoned_batch[0])
        if self.params["semantic"]:
            poison_images_list = copy.deepcopy(self.params["poison_images"])
            random.shuffle(poison_images_list)
            poison_images_test_list = copy.deepcopy(self.params["poison_images_test"])
            random.shuffle(poison_images_test_list)

        for pos in range(len(batch[0])):
            if pos < poisoned_len:
                if self.params["semantic"] and not self.params["edge_case"]:
                    if not evaluation:
                        poison_choice = poison_images_list[pos % len(self.params["poison_images"])]
                    else:
                        poison_choice = poison_images_test_list[pos % len(self.params["poison_images_test"])]
                    poisoned_batch[0][pos] = self.train_dataset[poison_choice][0]
                elif self.params["semantic"] and self.params["edge_case"]:
                    transform_edge_case = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                    if not evaluation:
                        poison_choice = random.choice(range(len(self.edge_case_train)))
                        poisoned_batch[0][pos] = transform_edge_case(self.edge_case_train[poison_choice])
                    else:
                        poison_choice = random.choice(range(len(self.edge_case_test)))
                        poisoned_batch[0][pos] = transform_edge_case(self.edge_case_test[poison_choice])

                elif (self.params["pixel_pattern"] and poisoned_pattern_choose != None):
                    if poisoned_pattern_choose==10:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose, blend_pattern=self.blend_pattern, blend_alpha=self.params["blend_alpha"])
                    elif poisoned_pattern_choose==1:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose)
                    elif poisoned_pattern_choose==20:
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose, evaluation, model_id=model_id)

                poisoned_batch[1][pos] = self.params["poison_label_swap"]
        
        return poisoned_batch

    def _local_watermarking_test_sub(self, test_data, model=None):
        if model == None:
            model = self.global_model

        model.eval()
        total_loss = 0
        dataset_size = 0
        correct = 0
        wm_label_correct = 0
        wm_label_sum = 0
        data_iterator = copy.deepcopy(test_data)

        for batch_id, batch in enumerate(data_iterator):
            # poisoned_batch, clean_batch= self._watermarking_batch_injection(batch, evaluation=True)

            data, targets = batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)

            output = model(data)
            total_loss += self.ceriterion(output, targets, reduction='sum').item() 
            pred = output.data.max(1)[1]

            if batch_id==0 and self.params["show_train_log"]:
                logger.info(f"watermarking pred: {pred}")
                logger.info(f"watermarking targets: {targets}")

            poisoned_label = self.params["poison_label_swap"]
            wm_label_targets = torch.ones_like(targets) * poisoned_label
            wm_label_index = targets.eq(wm_label_targets.data.view_as(targets))
            wm_label_sum += wm_label_index.cpu().sum().item()
            wm_label_correct += pred.eq(targets.data.view_as(pred))[wm_label_index.bool()].cpu().sum().item() 

            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            dataset_size += len(targets)
            
        watermark_acc = 100.0 *(float(correct) / float(dataset_size))
        wm_label_acc = 100.0 *(float(wm_label_correct) / float(wm_label_sum))
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, watermark_acc, wm_label_acc)

    def _local_test_sub(self, test_data, model=None, test_poisoned=False, poisoned_pattern_choose=None):

        if model == None:
            model = self.local_model

        model.eval()
        total_loss = 0
        correct = 0
        dataset_size = len(test_data.dataset)

        data_iterator = test_data

        for batch_id, batch in enumerate(data_iterator):
            if test_poisoned:
                poisoned_batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=True)
            else:
                poisoned_batch = copy.deepcopy(batch)

            data, targets = poisoned_batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)

            output = model(data)
            total_loss += self.ceriterion(output, targets, reduction='sum').item() 
            pred = output.data.max(1)[1]

            clean_batch = copy.deepcopy(batch)
            _,clean_targets = clean_batch
            clean_targets = clean_targets.cuda().detach().requires_grad_(False)

            # if batch_id==0 and test_watermarking:
            #     logger.info(f"watermarking preds are:{pred}")
            #     logger.info(f"watermarking target labels are:{targets}")

            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            
        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, acc)


    def local_test(self, model_id, test_data, round, poisoned_pattern_choose=None):

        loss, acc = self._local_test_sub(test_data, test_poisoned=False)
        logger.info(f"model:{model_id}, round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p= self._local_test_sub(test_data, test_poisoned=True, poisoned_pattern_choose=poisoned_pattern_choose)
        logger.info(f"model:{model_id}, round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")
        
        return True
