import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from participants.servers.AbstractServer import AbstractServer
from participants.clients.TDFedMaliciousClient import TDFedMaliciousCoordinator

import numpy as np
import logging
import time
import copy
import math
import json
import random
import models.resnet
import models.vgg
from utils.utils import save_model
from torch.autograd import Variable

logger = logging.getLogger("logger")

from utils.utils import add_trigger
from utils.utils import save_model

class IndicatorServer(AbstractServer):
    
    def __init__(self, params, current_time, train_dataset, open_set, 
                 blend_pattern, edge_case_train, edge_case_test, open_set_label=None):
        super(IndicatorServer, self).__init__(params, current_time)
        self.watermarking_rounds = [round for round in range(self.params["global_watermarking_start_round"],
                                                         self.params["global_watermarking_end_round"],
                                                         self.params["global_watermarking_round_interval"])] 

        ### add saved_models
        self.train_dataset=train_dataset
        self.open_set=open_set
        self.open_set_label=open_set_label
        self.blend_pattern = blend_pattern
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test

        self.poisoned_acc=[]
        self.clean_acc=[]
        self.no_detected_malicious = 0
        self.no_undetected_malicious = 0
        self.no_detected_benign = 0
        self.no_misclassified_benign = 0
        self.no_processed_malicious_clients = 0
        self.no_processed_benign_clients = 0
        self.VWM_detection_threshold = self.params["VWM_detection_threshold_initial"]

        self.wm_mu = self.params["watermarking_mu"]

        self._create_additional_model()
        self._loss_function()

        self.after_wm_injection_bn_stats_dict = dict()

    def _create_additional_model(self):
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

    def _select_clients(self, round):
        r"""
        randomly select participating clients for each round
        """
        adversary_list = [i for i in range(self.params["no_of_adversaries"])] \
                            if round in self.poisoned_rounds else []

        selected_clients = random.sample(range(self.params["no_of_total_participants"]), \
                self.params["no_of_participants_per_round"]) \
                if round not in self.poisoned_rounds else \
                adversary_list + random.sample(range(self.params["no_of_adversaries"], self.params["no_of_total_participants"]), \
                self.params["no_of_participants_per_round"]-self.params["no_of_adversaries"])
        return selected_clients, adversary_list

    def aggregation(self, weight_accumulator, aggregated_model_id):
        r"""
        aggregate all the updates model to generate a new global model
        """
        no_of_participants_this_round = sum(aggregated_model_id)
        for name, data in self.global_model.state_dict().items():
            # update_per_layer = weight_accumulator[name] * \
            #             (self.params["eta"] / self.params["no_of_participants_per_round"])
            update_per_layer = weight_accumulator[name] * \
                        (self.params["eta"] / no_of_participants_this_round)

            data = data.float()
            data.add_(update_per_layer)
        
        return True

    def _check_norm(self, local_client, round, model_id):
        r"""
        Clip the local model to agreed bound
        """
        params_list = []
        for name, param in local_client.local_model.named_parameters():
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))

        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list)
        logger.info(f"round:{round}, local model {model_id} | l2_norm: {l2_norm}")

        return True

    def _norm_clip(self, local_model_state_dict, clip_value):
        r"""
        Clip the local model to agreed bound
        """
        params_list = []
        for name, param in local_model_state_dict.items():
            if "running" in name or "num_batches_tracked" in name:
                continue
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))

        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list)

        scale = max(1.0, float(torch.abs(l2_norm / clip_value)))

        if self.params["norm_clip"]:
            for name, data in local_model_state_dict.items():
                if "running" in name or "num_batches_tracked" in name:
                    continue
                new_value = self.global_model.state_dict()[name] + (local_model_state_dict[name] - self.global_model.state_dict()[name])/scale
                local_model_state_dict[name].copy_(new_value)

        return local_model_state_dict
    
    def _indicator(self, local_model_state_dict, wm_data):

        benign_client = []
        label_inds = []
        label_acc_ws = []
        for ind, model_state_dict in enumerate(local_model_state_dict):

            self.check_model.copy_params(self.global_model.state_dict())
            for name, data in model_state_dict.items():
                if "num_batches_tracked" in name:
                    continue

                if "running" in name:
                    if self.params["replace_original_bn"]:
                        new_value = self.after_wm_injection_bn_stats_dict[name]
                    else:
                        continue
                else:
                    new_value = data.clone().detach()

                self.check_model.state_dict()[name].copy_(new_value)

            wm_copy_data = copy.deepcopy(wm_data)
            _, _, label_acc_w, label_ind, _, _ \
                = self._global_watermarking_test_sub(test_data=wm_copy_data, model=self.check_model)
            
            label_inds.append(label_ind)
            label_acc_ws.append(label_acc_w)

            if label_acc_w < self.VWM_detection_threshold: 
                benign_client.append(ind)
        
        logger.info(f"label ind:{label_inds}")
        logger.info(f"label acc wm:{label_acc_ws}") 
        return benign_client

    def _check_is_poisoned_model(self, local_client, wm_data, test_data):
        r"""
        Clip the local model to agreed bound
        """
        params_list = []
        for name, param in local_client.local_model.named_parameters():
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))

        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list)

        self.check_model.copy_params(self.global_model.state_dict())
        for name, data in local_client.local_model.named_parameters():
            new_value = data.clone().detach()
            self.check_model.state_dict()[name].copy_(new_value)
        wm_copy_data = copy.deepcopy(wm_data)
        _, _, label_acc_w, label_ind, label_acc_list, wm_pred_distrib = self._global_watermarking_test_sub(test_data=wm_copy_data, model=self.check_model)

        logger.info(f"target label ({label_ind}) wm acc:{label_acc_w}")

        if label_acc_w >= self.VWM_detection_threshold: 
            return True
        else:
            return False


    def local_data_distrib(self, train_data):
        ### Initialization
        distrib_dict=dict()
        no_class = 100 if self.params["dataset"].upper()=="CIFAR100" else 10 
        for label in range(no_class):
            distrib_dict[label]=0
        
        ### count the class distribution
        for batch_id, batch in enumerate(train_data):
            _, targets = batch
            for target in targets:
                distrib_dict[int(target.item())] += 1
        ### count sum
        sum_no = 0
        for key, value in distrib_dict.items():
            sum_no += value

        ### count percentage
        percentage_dict=dict()
        for key,value in distrib_dict.items():
            percentage_dict[key] = round(value/sum_no, 2)

        return distrib_dict, percentage_dict, sum_no

    def batch_label_distrib(self, targets):
        distrib_dict=dict()
        no_class = 100 if self.params["dataset"].upper()=="CIFAR100" else 10
        sum_no = 0
        for label in targets:
            if not distrib_dict.get(label, False):
                distrib_dict[label] = 1
            else:
                distrib_dict[label] += 1

            sum_no+=1

        percentage_dict=dict()
        for key,value in distrib_dict.items():
            percentage_dict[key] = round(value/sum_no, 2)

        return distrib_dict, percentage_dict, sum_no

    def _cos_sim(self, client, target_params_variables):
        model_list = []
        poison_dir_list = []
        for key, value in client.local_model.named_parameters():
           model_list.append(value.view(-1))
           poison_dir_list.append(target_params_variables[key].view(-1))

        model_tensor = torch.cat(model_list).cuda()
        poison_dir_tensor = torch.cat(poison_dir_list).cuda()
        cs = F.cosine_similarity(model_tensor, poison_dir_tensor, dim=0)
        return cs

    def _model_cos_sim(self, model, target_params_variables):
        model_list = []
        poison_dir_list = []
        for key, value in model.named_parameters():
           model_list.append(value.view(-1))
           poison_dir_list.append(target_params_variables[key].view(-1))

        model_tensor = torch.cat(model_list).cuda()
        poison_dir_tensor = torch.cat(poison_dir_list).cuda()
        cs = F.cosine_similarity(model_tensor, poison_dir_tensor, dim=0)
        return cs
    
    def broadcast_upload(self, round, local_benign_client, local_malicious_client, train_dataloader, test_dataloader, global_dataloader, poison_train_dataloader):
        r"""
        Server broadcasts the global model to all participants.
        Every participants train its our local model and upload the weight difference to the server.
        The server then aggregate the changes in the weight_accumulator and return it.
        """
        ### Log info
        logger.info(f"Training on global round {round} begins")
        ood_data = copy.deepcopy(self.open_set)
        if self.params["show_local_test_log"]:
            distrib_dict, percentage_dict, sum_no = self.local_data_distrib(ood_data)
            logger.info(f"class distribution for ood data, total no:{sum_no}")
            logger.info(f"{distrib_dict}")
            logger.info(f"{percentage_dict}")
        
        ### Count adversaries in one global round
        current_no_of_adversaries = 0
        selected_clients, adversary_list= self._select_clients(round)
        for client_id in selected_clients:
            if client_id in adversary_list:
                current_no_of_adversaries += 1
        logger.info(f"There are {current_no_of_adversaries} adversaries in the training for round {round}")

        ### Initialize the accumulator for all participants
        weight_accumulator = dict()
        for name, data in self.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(data)

        ### Initialize to calculate the distance between updates and global model
        target_params_variables = dict()
        for name, param in self.global_model.state_dict().items():
            target_params_variables[name] = param.clone()

        local_model_state_dict = []
        local_norm_list = []
        for enum_id, model_id in enumerate(selected_clients):
            logger.info(f" ")

            if model_id in adversary_list:
                client = local_malicious_client
                client_train_data = poison_train_dataloader
                # client_train_data = train_dataloader[model_id]
            else:
                client = local_benign_client
                client_train_data = train_dataloader[model_id]
           
            ### count class distribution info
            if self.params["show_local_test_log"]:
                distrib_dict, percentage_dict, sum_no = self.local_data_distrib(client_train_data)
                logger.info(f"class distribution for model {model_id}, total no:{sum_no}")
                logger.info(f"{distrib_dict}")
                logger.info(f"{percentage_dict}")
            
            ### copy global model
            client.local_model.copy_params(self.global_model.state_dict())
            
            ### set requires_grad to True
            for name, params in client.local_model.named_parameters():
                params.requires_grad = True

            client.local_model.train()
            start_time = time.time()
            client.local_training(
                                 train_data = client_train_data, 
                                 target_params_variables = target_params_variables,
                                 test_data = test_dataloader,
                                 global_data = global_dataloader,
                                 is_log_train = self.params["show_train_log"],
                                 poisoned_pattern_choose = self.params["poisoned_pattern_choose"],
                                 round = round, model_id=model_id
                                  )

            logger.info(f"local training for model {model_id} finishes in {time.time()-start_time} sec")

            if self.params["show_local_test_log"] or model_id==0:

                logger.info(f"BEFORE clipping:")
                client.local_test(round=round, model_id=model_id, test_data=test_dataloader, poisoned_pattern_choose=self.params["poisoned_pattern_choose"])

                wm_data = copy.deepcopy(self.open_set)
                loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=client.local_model)
                logger.info(f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")
                logger.info(f" ")

            ### Clip the parameters norm to the agreed bound
            self._check_norm(local_client=client, round=round, model_id=model_id)
            norm = self._model_dist_norm(model=client.local_model, 
                                            target_params=target_params_variables)
            local_norm_list.append(norm)

            if self.params["show_local_test_log"] or model_id==0:
                logger.info(f"AFTER clipping:")
                client.local_test(round=round, model_id=model_id, test_data=test_dataloader, poisoned_pattern_choose=self.params["poisoned_pattern_choose"])

                wm_data = copy.deepcopy(self.open_set)
                loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=client.local_model)
                logger.info(f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")

            cs = self._cos_sim(client=client, target_params_variables=target_params_variables)
            logger.info(f"cosine similarity between model {model_id} and global model is {cs}")

            local_model_state_dict_sub = dict()
            for name, param in client.local_model.state_dict().items():
                local_model_state_dict_sub[name] = param.clone().detach()
            local_model_state_dict.append(local_model_state_dict_sub)

        wm_data = copy.deepcopy(self.open_set)
        benign_client = self._indicator(local_model_state_dict=local_model_state_dict,
                                        wm_data=wm_data)
        clip_value = np.median(local_norm_list) \
                    if not self.params["fix_nc_bound"] else self.params["nc_bound"]
        logger.info(f" ")
        if self.params["norm_clip"]:
            logger.info(f"Norm clip: clipped value is: {clip_value}")
        else:
            logger.info(f"Norm clip: dont clip in this round")
        aggregated_model_id = [0]*len(local_model_state_dict)
        ### Updates the weight accumulator
        for ind in benign_client:
            aggregated_model_id[ind]=1

            local_model_state_dict_clipped = self._norm_clip(local_model_state_dict[ind], clip_value)
            for name, param in local_model_state_dict[ind].items():
                if "num_batches_tracked" in name:
                    continue
                weight_accumulator[name].add_(local_model_state_dict_clipped[name]-self.global_model.state_dict()[name])

        for ind, model_id in enumerate(selected_clients):
            if ind in adversary_list:
                if aggregated_model_id[ind] == 0:
                    self.no_detected_malicious+=1
                else:
                    self.no_undetected_malicious+=1
            else:
                if aggregated_model_id[ind] == 0:
                    self.no_misclassified_benign+=1
                else:
                    self.no_detected_benign+=1

        self.no_processed_malicious_clients +=  len(adversary_list)
        self.no_processed_benign_clients +=  len(selected_clients) - len(adversary_list)
        logger.info(f"aggregated_model:{aggregated_model_id}")
        logger.info(f"correctly detected malicious clients:{self.no_detected_malicious}/{self.no_processed_malicious_clients}, \
                    undetected malicious clients:{self.no_undetected_malicious}/{self.no_processed_malicious_clients}")
        logger.info(f"correctly detected benign clients:{self.no_detected_benign}/{self.no_processed_benign_clients}, \
                    misclassified benign clients:{self.no_misclassified_benign}/{self.no_processed_benign_clients}")
        logger.info(f"current VWM detection threshold: {self.VWM_detection_threshold}")

        if self.VWM_detection_threshold > self.params["VWM_detection_threshold_min"]:
            self.VWM_detection_threshold -= self.params["VWM_detection_threshold_minus_on"]

        return weight_accumulator, aggregated_model_id

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
        
        return poisoned_batch, original_batch
    
    def _global_test_sub(self, test_data, model=None, test_poisoned=False, poisoned_pattern_choose=None):
        r"""
        test benign acc on global model
        """
        if model == None:
            model = self.global_model
    
        model.eval()
        total_loss = 0
        correct = 0

        dataset_size = len(test_data.dataset)
        
        data_iterator = test_data

        for batch_id, batch in enumerate(data_iterator):
            if test_poisoned:
                test_batch, original_batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=True)
            else:
                test_batch = copy.deepcopy(batch)
                original_batch = copy.deepcopy(batch)

            data, targets = test_batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)

            _, original_targets = original_batch
            original_targets = original_targets.cuda().detach().requires_grad_(False)

            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item() 

            pred = output.data.max(1)[1]

            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        model.train()
        if test_poisoned:self.poisoned_acc.append(acc)
        else:self.clean_acc.append(acc)
        return (total_l, acc)

    def _global_watermarking_test_sub(self, test_data, model=None):
        if model == None:
            model = self.global_model

        model.eval()
        total_loss = 0
        dataset_size = 0
        correct = 0
        wm_label_correct = 0
        wm_label_sum = 0
        data_iterator = test_data

        wm_label_sum_list = [0 for i in range(self.params["class_num"])]
        wm_label_correct_list = [0 for i in range(self.params["class_num"])]
        wm_label_acc_list = [0 for i in range(self.params["class_num"])]
        wm_label_dict = dict()
        for i in range(self.params["class_num"]):
            wm_label_dict[i] = 0

        for batch_id, batch in enumerate(data_iterator):

            data, targets = batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)

            output = model(data)
            total_loss += self.ceriterion(output, targets, reduction='sum').item() 
            pred = output.data.max(1)[1]

            if batch_id==0 and model != None and self.params["show_train_log"]:
                logger.info(f"watermarking targets:{targets}")
                logger.info(f"watermarking pred :{pred}")
            
            for pred_item in pred:
                wm_label_dict[pred_item.item()]+=1

            # poisoned_label = self.params["poison_label_swap"]
            for target_label in range(self.params["class_num"]):
                wm_label_targets = torch.ones_like(targets) * target_label
                wm_label_index = targets.eq(wm_label_targets.data.view_as(targets))

                wm_label_sum_list[target_label] += wm_label_index.cpu().sum().item()
                wm_label_correct_list[target_label] += pred.eq(targets.data.view_as(pred))[wm_label_index.bool()].cpu().sum().item() 

            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
            dataset_size += len(targets)
            
        watermark_acc = 100.0 *(float(correct) / float(dataset_size))
        for i in range(self.params["class_num"]):
            wm_label_dict[i] = round(wm_label_dict[i]/dataset_size,2)
        for target_label in range(self.params["class_num"]):
            wm_label_acc_list[target_label] = round(100.0 * (float(wm_label_correct_list[target_label]) / float(wm_label_sum_list[target_label])), 2)

        # wm_label_acc = 100.0 * (float(wm_label_correct) / float(wm_label_sum))
        wm_label_acc = max(wm_label_acc_list)
        wm_index_label = wm_label_acc_list.index(wm_label_acc)
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, watermark_acc, wm_label_acc, wm_index_label, wm_label_acc_list, wm_label_dict)

    def global_test(self, test_data, round, poisoned_pattern_choose=None, model=None):
        r"""
        global test to show test acc/loss for different tasks
        """
        loss, acc = self._global_test_sub(test_data, test_poisoned = False, model=model)
        logger.info(f"global model on round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p = self._global_test_sub(test_data, test_poisoned = True, poisoned_pattern_choose=poisoned_pattern_choose, model=model)
        logger.info(f"global model on round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")

        return (acc, acc_p)
    
    def soft_cross_entropy(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum()/input.shape[0]

    def combined_cross_entropy(self, input, target):
        loss = 0.
        for ind, single_target in enumerate(target):
            if 1 in single_target:
                single_input = input[ind].clone().unsqueeze(0)
                single_target = target[ind].clone().unsqueeze(0)
                loss += self.soft_cross_entropy(single_input, single_target).item()
            else:
                single_input = input[ind].clone().unsqueeze(0)
                single_output = target[ind].clone().unsqueeze(0)
                loss += (1 - nn.functional.cosine_similarity(single_input, single_target, dim=1))*50
   
        # logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return loss/input.shape[0]

    def ceriterion_build(self, input, target, soft_label=False, reduction=None):
        if soft_label:
            loss = self.combined_cross_entropy(input, target)
        else:
            loss = nn.functional.cross_entropy(input, target, reduction=reduction)
        
        return loss

    def _loss_function(self):
        self.ceriterion = self.ceriterion_build
        return True

    def _optimizer(self, round, model):
        lr = self.params["global_lr_init"] + \
            (round - self.params["start_round"]) * self.params["global_lr_add_by"] #0.00008
        if lr>self.params["global_lr_max"]:
            lr = self.params["global_lr_max"]
        momentum = self.params["global_momentum"] 
        weight_decay = self.params["global_weight_decay"] 

        logger.info(f"indicator lr:{lr}")
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
        return True

    def _scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                 milestones=self.params['global_milestones'],
                                                 gamma=self.params['global_lr_gamma'])
        return True

    def _projection(self, target_params_variables):
        model_norm = self._model_dist_norm(self.global_model, target_params_variables)
        if model_norm > self.params["global_projection_norm"] and self.params["global_is_projection_grad"]:
            norm_scale = self.params["global_projection_norm"] / model_norm
            for name, param in self.global_model.named_parameters():
                clipped_difference = norm_scale * (
                        param.data - target_params_variables[name])
                param.data.copy_(target_params_variables[name]+clipped_difference)
        return True

    def _model_norm(self, model):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

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
                # trans = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                random_mask = torch.randn(self.open_set[index[pos]][0].shape) \
                            * self.params["ood_noise_scale"]
                # poisoned_batch[0][pos] = 1*self.open_set[index[pos]][0] + 0*random_mask
                poisoned_batch[0][pos] = (1-self.params["ood_noise_percent"])*\
                    self.open_set[index[pos]][0] + self.params["ood_noise_percent"]*random_mask
                if self.params["noise_label_fixed"]:
                    poisoned_batch[1][pos]=self.open_set_label[index[pos]]
                else:
                    poisoned_batch[1][pos]=random.randint(0,self.params["class_num"]-1)

        return poisoned_batch, batch
    
    def get_one_vec(self, model, variable=False):
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            size += layer.view(-1).shape[0]
        if variable:
            sum_var = Variable(torch.cuda.FloatTensor(size).fill_(0))
        else:
            sum_var = torch.cuda.FloatTensor(size).fill_(0)
        size = 0
        for name, layer in model.named_parameters():
            if name == 'decoder.weight':
                continue
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]
        return sum_var

    def _cos_sim_loss(self, model, target_params_variables):
        model_vec = self.get_one_vec(model, variable=True)
        target_vec_list=[]
        for name, _ in model.named_parameters():
            target_vec_list.append(target_params_variables[name].view(-1))
        target_vec = torch.cat(target_vec_list).cuda()
            
        target_var = Variable(target_vec, requires_grad=False)
        # target_vec.requires_grad = False
        cs_sim = torch.nn.functional.cosine_similarity(model_vec, target_var, dim=0)
        # cs_sim = cs_loss(model_vec, target_vec)
        loss = 1-cs_sim
        return 1e3*loss

    def _global_watermark_injection(self, watermark_data, global_data, test_data, target_params_variables, round=None, model=None, regulation_list=None):

        if model==None:
            model = self.global_model
        model.train()

        total_loss = 0
        global_datalist = list(global_data)
        poisoned_sample_list = list()
        self._loss_function()
        self._optimizer(round, model)
        self._scheduler()

        logger.info(f"wm_mu:{self.wm_mu}")

        retrain_no_times = self.params["global_retrain_no_times"]
        
        for internal_round in range(retrain_no_times):

            if internal_round%50==0:
                logger.info(f"global watermarking injection round:{internal_round}")
            data_iterator = copy.deepcopy(watermark_data)

            for batch_id, watermark_batch in enumerate(data_iterator):
                self.optimizer.zero_grad()
                wm_data, wm_targets = watermark_batch                
                wm_data = wm_data.cuda().detach().requires_grad_(False)
                wm_targets = wm_targets.cuda().detach().requires_grad_(False)

                global_batch = global_datalist[batch_id%len(global_datalist)]
                gl_data, gl_targets = global_batch
                gl_data = gl_data.cuda().detach().requires_grad_(False)
                gl_targets = gl_targets.cuda().detach().requires_grad_(False)

                data = wm_data
                targets = wm_targets

                output = model(data) 
                pred = output.data.max(1)[1]

                class_loss = nn.functional.cross_entropy(output, targets)
                distance_loss = self._model_dist_norm_var(model, target_params_variables)
                cos_sim_loss = self._cos_sim_loss(model, target_params_variables)
                if self.params["regulation_method"]=="l2_norm+neurotoxin":
                    loss = class_loss + (self.wm_mu/2) * distance_loss + self.params["watermarking_cs"] * cos_sim_loss
                elif self.params["regulation_method"]=="ewc":
                    loss = class_loss
                    for name, param in model.named_parameters():
                        fisher_params = regulation_list[name]
                        optpar = target_params_variables[name]
                        loss += (fisher_params * (optpar - param).pow(2)).sum() * self.params["ewc_lambda"]

                loss.backward()
                if regulation_list != None and self.params["regulation_method"]=="l2_norm+neurotoxin":
                    self._apply_grad_mask(model, regulation_list)
                self.optimizer.step()
                
                self._projection(target_params_variables)
                total_loss += loss.data

                if internal_round == retrain_no_times-1 and batch_id==0:
                    loss, acc = self._global_test_sub(test_data, test_poisoned=False, model=model)
                    logger.info(f"round:{internal_round} | benign acc:{acc}, benign loss:{loss}")

                    wm_data = copy.deepcopy(self.open_set)
                    loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=model)
                    logger.info(f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")

                    logger.info(f" ")

            self.scheduler.step()

        return poisoned_sample_list

    def save_bn_rs(self):
        param_dict = dict()
        for name, params in self.check_model.state_dict().items():
            if "running_mean" in name or "running_var" in name:
                param_dict[name] = params.clone().detach().cpu().numpy().tolist()

        filename = f"{self.folder_path}/check_model_bn_rs.txt"
        with open(filename, "w") as f:
            json.dump(param_dict, f)

    def _fisher_update(self, model, clean_data):
        model.train()
        model.zero_grad()
        for inputs, labels in clean_data:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            loss = nn.functional.cross_entropy(output, labels)
            loss.backward(retain_graph=True)
        
        fisher_dict = dict()
        for name, param in model.named_parameters():
            fisher_dict[name] = param.grad.data.clone().pow(2)
        
        return fisher_dict

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

        if self.params['aggregate_all_layer'] == 1:
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

    def _check_topk_grad_mask_percentage(self, mask_grad_list):

        grad_mask_percentage_dict = dict()
        count = 0
        for ind, grad_mask in enumerate(mask_grad_list):
            count+=sum(grad_mask.view(-1))

        for ind, (name, _) in enumerate(self.global_model.named_parameters()):
            grad_mask_percentage_dict[name] = sum(mask_grad_list[ind].view(-1)).item()/count.item()

        grad_mask_percentage_dict = dict(sorted(grad_mask_percentage_dict.items(),key=lambda x:x[1], reverse=True))

        return grad_mask_percentage_dict
    
    def _update_openset_label(self):
        ood_datalist = list(self.open_set)
        ood_datalist_shape = self.params["ood_data_sample_lens"]//self.params["ood_data_batch_size"] * self.params["ood_data_batch_size"] 
        assigned_labels = np.array([i for i in range(10)] * (ood_datalist_shape//10) + [i for i in range(ood_datalist_shape%10)])
        np.random.shuffle(assigned_labels)
        assigned_labels = assigned_labels.reshape(self.params["ood_data_sample_lens"]//self.params["ood_data_batch_size"], self.params["ood_data_batch_size"])
        for batch_id, batch in enumerate(ood_datalist):
            _, targets = batch
            for ind in range(len(targets)):
                targets[ind] = assigned_labels[batch_id][ind]
        ood_dataloader=iter(ood_datalist)
        return ood_dataloader

    def pre_process(self, global_data, test_data, round):

        wm_data = copy.deepcopy(self.open_set)
        loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=self.global_model)
        logger.info(f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")

        self.global_test(round=round, test_data=test_data, 
                        poisoned_pattern_choose=self.params["poisoned_pattern_choose"])
        logger.info(f" ")

        ### Initialize to calculate the distance between updates and global model
        if round in self.watermarking_rounds:
            target_params_variables = dict()
            for name, param in self.global_model.state_dict().items():
                target_params_variables[name] = param.clone()

            if self.params["show_local_test_log"]:
                distrib_dict, percentage_dict, sum_no = self.local_data_distrib(global_data)
                logger.info(f"class distribution for global data, total no:{sum_no}")
                logger.info(f"{distrib_dict}")
                logger.info(f"{percentage_dict}")

            if self.params["regulation_method"] == "l2_norm+neurotoxin":
                self.check_model.copy_params(self.global_model.state_dict())
                regulation_list = self._grad_mask_cv(model=self.check_model, clean_data=global_data, ratio=self.params["partial_watermarking_ratio"])
            elif self.params["regulation_method"] == "ewc":
                self.check_model.copy_params(self.global_model.state_dict())
                regulation_list = self._fisher_update(model=self.check_model, clean_data=global_data)

            before_wm_injection_bn_stats_dict = dict()
            for key, value in self.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    before_wm_injection_bn_stats_dict[key] = value.clone().detach()
            
            logger.info(f"benign inserting new watermarking")
            wm_data = copy.deepcopy(self.open_set)
            watermarking_sample_list = self._global_watermark_injection(watermark_data=wm_data,
                                             global_data=global_data, test_data=test_data, 
                                             target_params_variables=target_params_variables,
                                             model=self.global_model,
                                             regulation_list=regulation_list,
                                             round=round
                                             )

            watermarking_update_norm = self._model_dist_norm(self.global_model, target_params_variables)
            logger.info(f"watermarking update norm is :{watermarking_update_norm}")

            cs = self._model_cos_sim(model=self.global_model, target_params_variables=target_params_variables)
            logger.info(f"cosine similarity between old model and new global model is {cs}")

            wm_data = copy.deepcopy(self.open_set)
            loss_w, acc_w, label_acc_w, label_ind, _, _ = self._global_watermarking_test_sub(test_data=wm_data, model=self.global_model)
            logger.info(f"watermarking acc:{acc_w}, watermarking loss:{loss_w}, target label ({label_ind}) wm acc:{label_acc_w}")

            self.global_test(round=round, test_data=test_data)

            for key, value in self.global_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.after_wm_injection_bn_stats_dict[key] = value.clone().detach()

            self.check_model.copy_params(self.global_model.state_dict())
            for key, value in self.check_model.state_dict().items():
                if "running_mean" in key or "running_var" in key:
                    self.check_model.state_dict()[key].\
                        copy_(before_wm_injection_bn_stats_dict[key])
                    if self.params["replace_original_bn"]:
                        self.global_model.state_dict()[key].\
                            copy_(before_wm_injection_bn_stats_dict[key])
            logger.info(f"after replace wm bn with original bn:")
            self.global_test(round=round, test_data=test_data, model=self.check_model)

            logger.info(f" ")
        return True

    def post_process(self):
        return True