import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from participants.servers.AbstractServer import AbstractServer
import sklearn.metrics.pairwise as smp

import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

class RflbatServer(AbstractServer):
    
    def __init__(self, params, current_time, train_dataset, blend_pattern, 
                 edge_case_train, edge_case_test):
        super(RflbatServer, self).__init__(params, current_time)
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
    
    def _gap_statistics(self, data, num_sampling, K_max, n):
        num_cluster = 0
        data = np.reshape(data, (data.shape[0], -1))
        # Linear transformation
        data_c = np.ndarray(shape=data.shape)
        for i in range(data.shape[1]):
            data_c[:,i] = (data[:,i] - np.min(data[:,i])) / \
                 (np.max(data[:,i]) - np.min(data[:,i]))

        gap = []
        s = []
        for k in range(1, K_max + 1):
            k_means = KMeans(n_clusters=k, init='k-means++').fit(data_c)
            predicts = (k_means.labels_).tolist()
            centers = k_means.cluster_centers_
            v_k = 0
            for i in range(k):
                for predict in predicts:
                    if predict == i:
                        v_k += np.linalg.norm(centers[i] - \
                                 data_c[predicts.index(predict)])
            # perform clustering on fake data
            v_kb = []
            for _ in range(num_sampling):
                data_fake = []
                for i in range(n):
                    temp = np.ndarray(shape=(1,data.shape[1]))
                    for j in range(data.shape[1]):
                        temp[0][j] = random.uniform(0,1)
                    data_fake.append(temp[0])
                k_means_b = KMeans(n_clusters=k, init='k-means++').fit(data_fake)
                predicts_b = (k_means_b.labels_).tolist()
                centers_b = k_means_b.cluster_centers_
                v_kb_i = 0
                for i in range(k):
                    for predict in predicts_b:
                        if predict == i:
                            v_kb_i += np.linalg.norm(centers_b[i] - \
                                    data_fake[predicts_b.index(predict)])
                v_kb.append(v_kb_i)
            # gap for k
            v = 0
            for v_kb_i in v_kb:
                v += math.log(v_kb_i)
            v /= num_sampling
            gap.append(v - math.log(v_k))
            sd = 0
            for v_kb_i in v_kb:
                sd += (math.log(v_kb_i) - v)**2
            sd = math.sqrt(sd / num_sampling)
            s.append(sd * math.sqrt((1 + num_sampling) / num_sampling))

        # select smallest k
        for k in range(1, K_max + 1):
            print(gap[k - 1] - gap[k] + s[k])
            if k == K_max:
                num_cluster = K_max
                break
            if gap[k - 1] - gap[k] + s[k] > 0:
                num_cluster = k
                break

        return num_cluster

    def _rflbat(self, update_params):
        eps1 = self.params["rflbat_eps1"]
        eps2 = self.params["rflbat_eps2"]

        update_params_tensor = copy.deepcopy(update_params)
        for ind, update_param in enumerate(update_params):
            update_param = update_param.cpu().numpy().tolist()
            update_params[ind] = update_param
        
        update_params_tensor = torch.stack(update_params_tensor)
        U, S, V = torch.pca_lowrank(update_params_tensor)
        X_dr = torch.mm(update_params_tensor, V[:,:2]).cpu().numpy()
        
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i==j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
            eu_list.append(eu_sum)
        
        accept = []
        x1 = []

        for i in range(len(eu_list)):
            if eu_list[i] < eps1 * np.median(eu_list):
                accept.append(i)
                x1 = np.append(x1, X_dr[i])
        
        x1 = np.reshape(x1, (-1, X_dr.shape[1]))
        num_clusters = self._gap_statistics(x1, num_sampling=5, \
                                           K_max=9, n=len(x1))
        logger.info(f"num of selected clusters:{num_clusters}")
        k_means = KMeans(n_clusters=num_clusters, init="k-means++").fit(x1)
        predicts = k_means.labels_

        v_med = []
        for i in range(num_clusters):
            temp = []
            for j in range(len(predicts)):
                if predicts[j] == i:
                    temp.append(update_params[accept[j]])
            if len(temp) <= 1:
                v_med.append(1)
                continue
            v_med.append(np.median(np.average(smp\
                .cosine_similarity(temp), axis=1)))
        temp = []
        for i in range(len(accept)):
            if predicts[i] == v_med.index(min(v_med)):
                temp.append(accept[i])
        accept = temp

        # compute eu list again to exclude outliers
        temp = []
        for i in accept:
            temp.append(X_dr[i])
        X_dr = temp
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i==j:
                    continue
                eu_sum += np.linalg.norm(X_dr[i]-X_dr[j])
            eu_list.append(eu_sum)
        temp = []
        for i in range(len(eu_list)):
            if eu_list[i] < eps2 * np.median(eu_list):
                temp.append(accept[i])

        accept = temp
    
        return accept

    def broadcast_upload(self, round, local_benign_client, local_malicious_client, train_dataloader, test_dataloader, global_dataloader, poison_train_dataloader):

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
                                 global_data = global_dataloader,
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
 
            update_params_sub = []
            for name, param in client.local_model.named_parameters():
                update_params_value = param.detach().clone() - target_params_variables[name].detach().clone()
                update_params_sub.append(update_params_value.view(-1))
            update_params_sub = torch.cat(update_params_sub).cuda()
            update_params.append(update_params_sub)

            local_model_state_dict_sub = dict()
            for name, param in client.local_model.state_dict().items():
                local_model_state_dict_sub[name] = param.clone()
            local_model_state_dict.append(local_model_state_dict_sub)

        logger.info(f" ")
        benign_clients = self._rflbat(update_params=update_params)
        for ind, client_id in enumerate(benign_clients):
            aggregated_model_id[client_id]=1
            for name, param in local_model_state_dict[ind].items():
                if "num_batches_tracked" in name:
                    continue
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