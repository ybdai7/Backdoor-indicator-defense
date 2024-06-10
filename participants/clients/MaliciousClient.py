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

class MaliciousClient(AbstractClient):
    def __init__(self, params, train_dataset, blend_pattern, open_set, 
                 edge_case_train, edge_case_test, open_set_label=None):
        super(MaliciousClient, self).__init__(params)
        self.train_dataset = train_dataset
        sample_data, _ = self.train_dataset[1]
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test

        self.blend_pattern = blend_pattern
        self.open_set = open_set
        self.open_set_label = open_set_label
        self._create_check_model()
        self._loss_function()

    def _create_check_model(self):
        r"""
        create global model according to the uploaded params info,
        ATTENTION: VGG model does not support EMNIST task YET!
        """
        if "ResNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="CIFAR")
            elif self.params["dataset"].upper() == "CIFAR100":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=100, dataset="CIFAR")
            elif self.params["dataset"].upper() == "EMNIST":
                check_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="EMNIST")
        elif "VGG" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                check_model = getattr(models.vgg, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                check_model = getattr(models.vgg, self.params["model_type"])(num_classes=100)
        
        self.check_model = check_model.cuda()
        return True

    def soft_cross_entropy(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum()/input.shape[0]

    def combined_cross_entropy(self, input, target):
        loss = 1 - nn.functional.cosine_similarity(input, target, dim=1).item()
        # logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return loss/input.shape[0]

    def ceriterion_build(self, input, target, soft_label=False, reduction=None):
        if soft_label:
            loss = self.combined_cross_entropy(input, target)
        else:
            loss = nn.functional.cross_entropy(input, target, reduction=reduction)
        
        return loss

    def _loss_function(self):
        # self.ceriterion = self.ceriterion_build
        self.ceriterion = nn.functional.cross_entropy
        return True

    # def _loss_function(self):
    #     self.ceriterion = nn.functional.cross_entropy
    #
    #     return True

    def _optimizer(self, round, adaptive):
        if adaptive:
            lr = self.params["adaptive_attack_lr"]
        else:
            lr = self.params["poisoned_lr"]
        logger.info(f"malicious lr:{lr}")
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr,
                                    momentum=self.params["poisoned_momentum"], weight_decay=self.params["poisoned_weight_decay"])  
        return True

    def _scheduler(self, adaptive):
        if adaptive:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                 milestones=self.params['adaptive_malicious_milestones'],
                                                 gamma=self.params['adaptive_malicious_lr_gamma'])
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                 milestones=self.params['malicious_milestones'],
                                                 gamma=self.params['malicious_lr_gamma'])
            
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

    def _projection(self, target_params_variables):
        model_norm = self._model_dist_norm(self.local_model, target_params_variables)
        if self.params["show_train_log"]:
            logger.info(f"model dist is :{model_norm}")

        if model_norm > self.params["poisoned_projection_norm"] and self.params["poisoned_is_projection_grad"]:
            norm_scale = self.params["poisoned_projection_norm"] / model_norm
            for name, param in self.local_model.named_parameters():
                clipped_difference = norm_scale * (
                        param.data - target_params_variables[name])
                param.data.copy_(target_params_variables[name]+clipped_difference)

        return True

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

    def _grad_mask_cv(self, model, clean_data, ratio=None):
        """Generate a gradient mask based on the given dataset"""
        model.train()
        model.zero_grad()

        for internal_round in range(10):
            for inputs, labels in clean_data:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                loss = nn.functional.cross_entropy(output, labels)
                loss.backward(retain_graph=True)
        mask_grad_list = []

        if self.params['malicious_aggregate_all_layer'] == 1:
            grad_list = []
            grad_abs_sum_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_list.append(parms.grad.abs().view(-1))
                    grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

            grad_list = torch.cat(grad_list).cuda()
            if not isinstance(ratio, list):
                _, indices = torch.topk(-1 * grad_list, int(len(grad_list)*ratio))
                mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
                mask_flat_all_layer[indices] = 1.0

            else:
                left_ratio = ratio[0]
                right_ratio = ratio[1]
                _, left_indices = torch.topk(grad_list, int(len(grad_list)*left_ratio))
                _, right_indices = torch.topk(grad_list, int(len(grad_list)*right_ratio))
                mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
                mask_flat_all_layer[right_indices] = 1.0
                mask_flat_all_layer[left_indices] = 0.0


            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients_length = len(parms.grad.abs().view(-1))

                    mask_flat = mask_flat_all_layer[count:count + gradients_length ].cuda()
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                    count += gradients_length
                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                    percentage_mask_list.append(percentage_mask1)
                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))
                    k_layer += 1

        else:
            grad_abs_percentage_list = []
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    grad_res.append(parms.grad.view(-1))
                    l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().cuda())/float(len(parms.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            grad_flat = torch.cat(grad_res)

            percentage_mask_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    if ratio == 1.0:
                        _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                    else:

                        ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                        _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))

                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                    percentage_mask_list.append(percentage_mask1)
                    k_layer += 1

        model.zero_grad()
        return mask_grad_list
    
    def _apply_grad_mask(self, model, mask_grad_list):
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)

    def local_training(self, train_data, test_data, target_params_variables, is_log_train, poisoned_pattern_choose=None, round=None, model_id=None):
        total_loss = 0
        data_iterator = train_data
        self._loss_function()
        if self.params["adaptive_attack"]:
            self._optimizer(round, adaptive=True)
            self._scheduler(adaptive=True)
        else:
            self._optimizer(round, adaptive=False)
            self._scheduler(adaptive=False)
        mask_grad_list = self._grad_mask_cv(model=self.local_model, clean_data=train_data, 
                                            ratio=self.params["malicious_neurotoxin_ratio"])

        if self.params["adaptive_attack"]:
            retrain_no_times = self.params["adaptive_attack_round"] + self.params["poisoned_retrain_no_times"]
        else:
            retrain_no_times = self.params["poisoned_retrain_no_times"]

        for internal_round in range(retrain_no_times):
            logger.info(f"Malicious training: plr:{self.optimizer.state_dict()['param_groups'][0]['lr']}")
            for batch_id, batch in enumerate(data_iterator):
                self.optimizer.zero_grad()

                if self.params["adaptive_attack"] and internal_round < self.params["adaptive_attack_round"]:
                    batch = copy.deepcopy(batch)
                else:
                    batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=False, model_id=model_id)

                data, targets = batch
                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                output = self.local_model(data)
                # loss = self.ceriterion(output, targets)
                class_loss = self.ceriterion(output, targets)
                distance_loss = self._model_dist_norm_var(self.local_model, target_params_variables)
                loss = class_loss
                loss.backward()
                self._apply_grad_mask(self.local_model, mask_grad_list)
                self.optimizer.step()

                self._projection(target_params_variables)
                total_loss += loss.data

                # if batch_id % 10 == 0 and is_log_train:
                # if batch_id % 2 == 0 and is_log_train:
                # if batch_id == len(data_iterator)-1 and is_log_train:
                # if batch_id == len(data_iterator)-1 and internal_round == self.params["benign_retrain_no_times"]-1 and is_log_train:
                if is_log_train:

                    loss, acc = self._local_test_sub(test_data, test_poisoned=False, model=self.local_model)
                    logger.info(f"round:{internal_round} | benign acc:{acc}, benign loss:{loss}")

                    loss_p, acc_p = self._local_test_sub(test_data, test_poisoned=True, poisoned_pattern_choose=poisoned_pattern_choose, model=self.local_model)
                    logger.info(f"round:{internal_round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")
                    
                    wm_data=copy.deepcopy(self.open_set)
                    loss_w, acc_w, label_acc_w = self._local_watermarking_test_sub(wm_data, model=self.local_model)
                    logger.info(f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target class wm acc:{label_acc_w}")

                    logger.info(f" ")
                    
            if self.params["adaptive_attack"] and \
                internal_round==self.params["adaptive_attack_round"]-1:
                self._optimizer(round, adaptive=False)
                self._scheduler(adaptive=False)
            else:
                self.scheduler.step()

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
                        poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose, evaluation=evaluation, model_id=model_id)


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
