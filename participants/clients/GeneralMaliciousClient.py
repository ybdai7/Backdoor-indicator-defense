import torch
import torch.nn as nn
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

class GeneralMaliciousClient(AbstractClient):
    def __init__(self, params):
        super(GeneralMaliciousClient, self).__init__(params)

    def _loss_function(self):
        self.ceriterion = nn.functional.cross_entropy

        return True

    def _optimizer(self):
        self.optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.params["poisoned_lr"],
                                    momentum=self.params["poisoned_momentum"], weight_decay=self.params["poisoned_weight_decay"])  
        return True

    def _model_dist_norm(self, model, target_params):
        squared_sum = 0
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data - target_params[name].data, 2))
        return math.sqrt(squared_sum)

    def _projection(self, target_params_variables):
        model_norm = self._model_dist_norm(self.local_model, target_params_variables)
        if model_norm > self.params["poisoned_projection_norm"] and self.params["poisoned_is_projection_grad"]:
            norm_scale = self.params["poisoned_projection_norm"] / model_norm
            for name, param in self.local_model.named_parameters():
                clipped_difference = norm_scale * (
                        param.data - target_params_variables[name])
                param.data.copy_(target_params_variables[name]+clipped_difference)

        return True

    def local_training(self, train_data, test_data, target_params_variables, poisoned_pattern_choose=None):
        total_loss = 0
        data_iterator = train_data
        self._loss_function()
        self._optimizer()

        for internal_round in range(self.params["poisoned_retrain_no_times"]):
            for batch_id, batch in enumerate(data_iterator):
                self.optimizer.zero_grad()
                batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=False)
                data, targets = batch

                data = data.cuda().detach().requires_grad_(False)
                targets = targets.cuda().detach().requires_grad_(False)

                output = self.local_model(data)
                loss = self.ceriterion(output, targets)
                loss.backward()
                self.optimizer.step()

                self._projection(target_params_variables)
                total_loss += loss.data

                if batch_id % 3 == 0:
                    loss, acc = self._local_test_sub(test_data, round, test_poisoned=False)
                    logger.info(f"round:{internal_round} | benign acc:{acc}, benign loss:{loss}")

                    loss_p, acc_p = self._local_test_sub(test_data, round, test_poisoned=True, poisoned_pattern_choose=poisoned_pattern_choose)
                    logger.info(f"round:{internal_round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")
                    
    
    def _poisoned_batch_injection(self, batch, poisoned_pattern_choose=None, evaluation=False):
        r"""
        replace the poisoned batch with the oirginal batch
        """
        poisoned_batch = copy.deepcopy(batch)
        poisoned_len = self.params["poisoned_len"] if not evaluation else len(poisoned_batch[0])

        for pos in range(len(batch[0])):
            if pos < poisoned_len:
                if self.params["semantic"]:
                    if not evaluation:
                        poison_choice = random.choice(self.params["poison_images"])
                    else:
                        poison_choice = random.choice(self.params["poison_images_test"])
                    poisoned_batch[0][pos] = train_dataset[poison_choice][0]
                elif self.params["pixel_pattern"] and poisoned_pattern_choose != None:
                    poisoned_batch[0][pos] = add_trigger(poisoned_batch[0][pos], poisoned_pattern_choose)

                poisoned_batch[1][pos] = self.params["poison_label_swap"]
        
        return poisoned_batch

    def _local_test_sub(self, test_data, round, test_poisoned=False, poisoned_pattern_choose=None):
        
        self.local_model.eval()
        total_loss = 0
        correct = 0

        dataset_size = len(test_data.dataset)
        data_iterator = test_data

        for batch_id, batch in enumerate(data_iterator):
            if test_poisoned:
                batch = self._poisoned_batch_injection(batch, poisoned_pattern_choose, evaluation=True)
            else:
                batch = copy.deepcopy(batch)
            data, targets = batch
            data = data.cuda().detach().requires_grad_(False)
            targets = targets.cuda().detach().requires_grad_(False)

            output = self.local_model(data)
            total_loss += self.ceriterion(output, targets, reduction='sum').item() 
            pred = output.data.max(1)[1]
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        self.local_model.train()
        return (total_l, acc)

    def local_test(self, model_id, test_data, round, poisoned_pattern_choose=None):

        loss, acc = self._local_test_sub(test_data, round, test_poisoned=False)
        logger.info(f"model:{model_id}, round:{round} | benign acc:{acc}, benign loss:{loss}")

        loss_p, acc_p = self._local_test_sub(test_data, round, test_poisoned=True, poisoned_pattern_choose=poisoned_pattern_choose)
        logger.info(f"model:{model_id}, round:{round} | poisoned acc:{acc_p}, poisoned loss:{loss_p}")
        
        return True
