import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from participants.servers.AbstractServer import AbstractServer
from numpy import dot
from numpy.linalg import norm

import numpy as np
import random
import logging
import time
import copy
import math
import json
import hdbscan

import models.resnet
import models.vgg
from utils.utils import save_model
from utils.utils import NoiseDataset

logger = logging.getLogger("logger")

from utils.utils import add_trigger

class DeepsightServer(AbstractServer):
    
    def __init__(self, params, current_time, train_dataset, blend_pattern, 
                 edge_case_train, edge_case_test):
        super(DeepsightServer, self).__init__(params, current_time)
        self.train_dataset=train_dataset
        self.blend_pattern=blend_pattern
        self.edge_case_train = edge_case_train
        self.edge_case_test = edge_case_test

        self.no_detected_malicious = 0
        self.no_undetected_malicious = 0
        self.no_detected_benign = 0
        self.no_misclassified_benign = 0
        self.no_processed_malicious_clients = 0
        self.no_processed_benign_clients = 0
        self._create_check_model()

    def _create_check_model(self):
        r"""
        create global model according to the uploaded params info,
        ATTENTION: VGG model does not support EMNIST task YET!
        """
        if "ResNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                sub_local_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="CIFAR")
                sub_global_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="CIFAR")
            elif self.params["dataset"].upper() == "CIFAR100":
                sub_local_model = getattr(models.resnet, self.params["model_type"])(num_classes=100, dataset="CIFAR")
                sub_global_model = getattr(models.resnet, self.params["model_type"])(num_classes=100, dataset="CIFAR")
            elif self.params["dataset"].upper() == "EMNIST":
                sub_local_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="EMNIST")
                sub_global_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="EMNIST")
        elif "VGG" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                sub_local_model = getattr(models.vgg, self.params["model_type"])(num_classes=10)
                sub_global_model = getattr(models.vgg, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                sub_local_model = getattr(models.vgg, self.params["model_type"])(num_classes=100)
                sub_global_model = getattr(models.vgg, self.params["model_type"])(num_classes=100)
        
        self.sub_local_model = sub_local_model.cuda()
        self.sub_global_model = sub_global_model.cuda()
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
        no_of_participants_this_round = len(aggregated_model_id)
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * \
                        (self.params["eta"] / no_of_participants_this_round)
            
            data = data.float()
            data.add_(update_per_layer)
        return True
    
    def _norm_check(self, local_client, round, model_id):
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

    def _norm_clip(self, local_model_vector, clip_value):
        r"""
        Clip the local model to agreed bound
        """
        params_list = []
        for name, param in local_model_vector.items():
            diff_value = param - self.global_model.state_dict()[name]
            params_list.append(diff_value.view(-1))

        params_list = torch.cat(params_list)
        l2_norm = torch.norm(params_list)

        scale = max(1.0, float(torch.abs(l2_norm / clip_value)))

        if self.params["norm_clip"]:
            for name, data in local_model_vector.items():
                new_value = self.global_model.state_dict()[name] + (local_model_vector[name] - self.global_model.state_dict()[name])/scale
                local_model_vector[name].copy_(new_value)

        return local_model_vector

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
    
    def _dists_from_clust(self, clusters, N):
        pairwise_dists = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                pairwise_dists[i,j] = 0 if clusters[i] == clusters[j] else 1
        return pairwise_dists

    def _deepsight(self, local_model_state_dict):
        num_seeds = self.params["deepsight_num_seeds"]
        num_samples = self.params["deepsight_num_samples"]
        num_channel = self.params["deepsight_num_channel"]
        tau = self.params["deepsight_tau"]

        last_layer_name = "linear" if "ResNet" in self.params["model_type"] else "classifier"
        if self.params["dataset"].upper() == "CIFAR10":
            num_classes = 10
            dim = 32
        elif self.params["dataset"].upper() == "CIFAR100":
            num_classes = 100
            dim = 32
        elif self.params["dataset"].upper() == "EMNIST":
            num_classes = 10
            dim = 28

        ### computing NEUPs and TEs
        TEs, NEUPs, ed = [], [], []
        for enu_id, local_model_state_dict_sub in enumerate(local_model_state_dict):
            ### get updated params norm
            squared_sum = 0
            for name, value in local_model_state_dict_sub.items():
                if "tracked" in name or "running" in name:
                    continue
                squared_sum += torch.sum(torch.pow(value-self.global_model.state_dict()[name], 2)).item()
            update_norm = math.sqrt(squared_sum)
            ed = np.append(ed, update_norm)

            diff_bias = local_model_state_dict_sub[f"{last_layer_name}.bias"] \
                - self.global_model.state_dict()[f"{last_layer_name}.bias"]
            diff_weight = local_model_state_dict_sub[f"{last_layer_name}.weight"] \
                - self.global_model.state_dict()[f"{last_layer_name}.weight"]
            
            UPs = abs(diff_bias.cpu().numpy()) + np.sum(abs(diff_weight.cpu().numpy()), axis=1)
            NEUP = UPs**2/np.sum(UPs**2)
            TE = 0

            for j in NEUP:
                if j>= (1/num_classes)*np.max(NEUP):
                    TE += 1
            NEUPs = np.append(NEUPs, NEUP)
            TEs.append(TE)
        
        logger.info("Deepsight: Finish cauculating TE")
        labels = []
        for i in TEs:
            if i >= np.median(TEs)/2:
                labels.append(0)
            else:
                labels.append(1)
        logger.info(f"computed TEs:{TEs}")
        logger.info(f"computed labels:{labels}")
        
        ### computing DDifs
        DDifs = []
        for i, seed in enumerate(range(num_seeds)):
            torch.manual_seed(seed)
            dataset = NoiseDataset([num_channel, dim, dim], num_samples)
            loader = torch.utils.data.DataLoader(dataset, self.params["deepsight_batch_size"], shuffle=False)

            for enu_id, local_model_state_dict_sub in enumerate(local_model_state_dict):
                self.sub_local_model.copy_params(local_model_state_dict_sub)
                self.sub_global_model.copy_params(self.global_model.state_dict())
                self.sub_local_model.eval()
                self.sub_global_model.eval()

                DDif = torch.zeros(num_classes).cuda()
                for y in loader:
                    x,_ = y 
                    x = x.cuda()
                    with torch.no_grad():
                        output_local = self.sub_local_model(x)
                        output_global = self.sub_global_model(x)

                        output_local = torch.softmax(output_local, dim=1)
                        output_global = torch.softmax(output_global, dim=1)

                    temp = torch.div(output_local, output_global+1e-30) # avoid zero-value
                    temp = torch.sum(temp, dim=0)
                    DDif.add_(temp)

                DDif /= num_samples
                DDifs = np.append(DDifs, DDif.cpu().numpy())

        DDifs = np.reshape(DDifs, (num_seeds, len(local_model_state_dict), -1))
        logger.info("Deepsight: Finish cauculating DDifs")

        ### compute cosine distance
        bias_name = f"{last_layer_name}.bias"
        cosine_distance = np.zeros((len(local_model_state_dict), len(local_model_state_dict)))
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()

        for ind_0, local_model_state_dict_sub_0 in enumerate(local_model_state_dict):
            for ind_1, local_model_state_dict_sub_1 in enumerate(local_model_state_dict):
                update_0 =  local_model_state_dict_sub_0[bias_name] \
                    - self.global_model.state_dict()[bias_name]
                update_1 =  local_model_state_dict_sub_1[bias_name]\
                    - self.global_model.state_dict()[bias_name]
                # cosine_distance[ind_0, ind_1] = 1.0 - dot(update_0, update_1)/(norm(update_0)*norm(update_1))
                cosine_distance[ind_0, ind_1] = 1.0 - cos(update_0, update_1)
        logger.info("Deepsight: Finish cauculating cosine distance")

        # classification
        cosine_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(cosine_distance)
        logger.info(f"cosine cluster:{cosine_clusters}")
        cosine_cluster_dists = self._dists_from_clust(cosine_clusters, len(local_model_state_dict))

        NEUPs = np.reshape(NEUPs, (len(local_model_state_dict),self.params["class_num"]))
        neup_clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(NEUPs)
        logger.info(f"neup cluster:{neup_clusters}")
        neup_cluster_dists = self._dists_from_clust(neup_clusters, len(local_model_state_dict))

        ddif_clusters, ddif_cluster_dists = [],[]
        for i in range(num_seeds):
            DDifs[i] = np.reshape(DDifs[i], (len(local_model_state_dict),self.params["class_num"]))
            ddif_cluster_i = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(DDifs[i])
            logger.info(f"ddif cluster:{ddif_cluster_i}")
            ddif_cluster_dists = np.append(ddif_cluster_dists,
                self._dists_from_clust(ddif_cluster_i, len(local_model_state_dict)))

        merged_ddif_cluster_dists = np.average(ddif_cluster_dists, axis=0)
        
        merged_distances = np.mean([merged_ddif_cluster_dists,
                                    neup_cluster_dists,
                                    cosine_cluster_dists], axis=0)

        clusters = hdbscan.HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(merged_distances)
        logger.info(f"cluster label:{clusters}")

        final_clusters = np.asarray(clusters)
        cluster_list = np.unique(final_clusters)
        benign_client = []
        labels = np.asarray(labels)
        for cluster in cluster_list:
            if cluster == -1:
                indexes = np.argwhere(final_clusters==cluster).flatten()
                for i in indexes:
                    if labels[i] == 1:
                        continue
                    else:
                        benign_client.append(i)
            else:
                indexes = np.argwhere(final_clusters==cluster).flatten()
                amount_of_suspicious = np.sum(labels[indexes])/len(indexes)
                if amount_of_suspicious < tau:
                    for idx in indexes:
                        benign_client.append(idx)

        # Aggregate and norm-clipping
        clip_value = np.median(ed)

        return benign_client, clip_value

    def broadcast_upload(self, round, local_benign_client, local_malicious_client, train_dataloader, test_dataloader, poison_train_dataloader):

        r"""
        Server broadcasts the global model to all participants.
        Every participants train its our local model and upload the weight difference to the server.
        The server then aggregate the changes in the weight_accumulator and return it.
        """
        ### Log info
        logger.info(f"Training on global round {round} begins")
            
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

        ### Start training for each participating local client
        aggregated_model_id = [0]*self.params["no_of_participants_per_round"]

        local_model_vector = []
        local_model_named_param = []
        update_params = []
        local_model_state_dict = []
        for model_id in selected_clients:
            logger.info(f" ")
            if model_id in adversary_list:
                client = local_malicious_client
                client_train_data = poison_train_dataloader
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
                                 is_log_train = self.params["show_train_log"],
                                 poisoned_pattern_choose = self.params["poisoned_pattern_choose"],
                                 round=round, model_id=model_id
                                  )


            logger.info(f"local training for model {model_id} finishes in {time.time()-start_time} sec")

            if model_id==0:
                client.local_test(round=round, model_id=model_id, test_data=test_dataloader, poisoned_pattern_choose=self.params["poisoned_pattern_choose"])
                logger.info(f" ")

            ### Clip the parameters norm to the agreed bound
            self._norm_check(local_client=client, round=round, model_id=model_id)

            cs = self._cos_sim(client=client, target_params_variables=target_params_variables)
            logger.info(f"cosine similarity between model {model_id} and global model is {cs}")
 
            local_model_state_dict_sub = dict()
            for name, param in client.local_model.state_dict().items():
                local_model_state_dict_sub[name] = param.clone().detach()
            local_model_state_dict.append(local_model_state_dict_sub)

            local_model_named_param_sub = dict()
            for name, param in client.local_model.named_parameters():
                local_model_named_param_sub[name] = param.clone().detach()
            local_model_named_param.append(local_model_named_param_sub)

            ### Updates the weight accumulator
            # for name, param in client.local_model.state_dict().items():
            #     weight_accumulator[name].add_(param -
            #     self.global_model.state_dict()[name])

        logger.info(f" ")
        benign_client, clip_value = self._deepsight(local_model_state_dict=local_model_state_dict)
        logger.info(f"benign clients are:{benign_client}, and clip value is: {clip_value}")
        for ind in benign_client:
            aggregated_model_id[ind]=1
            local_model_named_param_clipped = self._norm_clip(local_model_named_param[ind], clip_value)
            for name, param in local_model_state_dict[ind].items():
                if name in local_model_named_param_clipped.keys():
                    weight_accumulator[name].add_(local_model_named_param_clipped[name]-self.global_model.state_dict()[name])
                else:
                    weight_accumulator[name].add_(param - self.global_model.state_dict()[name])
        
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
                batch, original_batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=True)
            else:
                batch = copy.deepcopy(batch)
                original_batch = copy.deepcopy(batch)

            data, targets = batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)

            _, original_targets = original_batch
            original_targets = original_targets.cuda().detach().requires_grad_(False)

            output = model(data)
            total_loss += nn.functional.cross_entropy(output, targets, reduction='sum').item() 
            pred = output.data.max(1)[1]
            # if test_watermarking and batch_id==0:
            #     logger.info(f"targets:{targets}")
            #     logger.info(f"original targets:{original_targets}")
            #     logger.info(f"pred:{pred}")

            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        model.train()
        return (total_l, acc)

    def global_test(self, test_data, round, poisoned_pattern_choose=None):
        r"""
        global test to show test acc/loss for different tasks
        """
        loss, acc = self._global_test_sub(test_data, test_poisoned = False)
        logger.info(f"global model on round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p = self._global_test_sub(test_data, test_poisoned = True, poisoned_pattern_choose=poisoned_pattern_choose)
        logger.info(f"global model on round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")

        return (acc, acc_p)
    
    def pre_process(self, *args, **kwargs):
        return True

    def post_process(self):
        return True