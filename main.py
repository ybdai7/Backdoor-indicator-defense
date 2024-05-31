import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import time
import argparse
import yaml
import random
import datetime
import logging

from participants.servers.IndicatorServer import IndicatorServer
from participants.servers.FlameServer import FlameServer
from participants.servers.NodefenseServer import NodefenseServer
from participants.servers.DeepsightServer import DeepsightServer
from participants.servers.FoolsgoldServer import FoolsgoldServer
from participants.servers.RflbatServer import RflbatServer
from participants.servers.MultikrumServer import MultikrumServer

from participants.clients.BenignClient import BenignClient
from participants.clients.FedProxBenignClient import FedProxBenignClient
from participants.clients.MaliciousClient import MaliciousClient
from participants.clients.ChameleonMaliciousClient import ChameleonMaliciousClient

from dataloader.WMFLDataloader import WMFLDataloader
from utils.utils import save_model
from utils.utils import plot_poisoned_acc

logger = logging.getLogger("logger")

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# np.random.seed(0)
# random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="utils/params_pixel_WM_verifying_dyb.yaml")
    parser.add_argument("--GPU_id", default="0", type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_id
    with open(f"./{args.params}", "r") as f:
        params_loaded = yaml.safe_load(f)
    params_loaded.update(vars(args))

    # CIFAR10
    # params_loaded['dataset']="CIFAR10"
    # params_loaded['class_num']=10

    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    dataloader = WMFLDataloader(params=params_loaded)
    # generate blend pattern
    sample_data, _ = dataloader.train_dataset[1]
    channel, height, width = sample_data.shape
    blend_pattern = (torch.rand(sample_data.shape) - 0.5) * 2

    if dataloader.params["defense_method"].lower()=="nodefense":
        server = NodefenseServer(params=params_loaded, current_time=current_time, train_dataset=dataloader.train_dataset, 
                            blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                            edge_case_test=dataloader.edge_poison_test)
    elif dataloader.params["defense_method"].lower()=="indicator":
        server = IndicatorServer(params=params_loaded, current_time=current_time, train_dataset=dataloader.train_dataset, 
                                open_set=dataloader.ood_data, blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                                edge_case_test=dataloader.edge_poison_test)
    elif dataloader.params["defense_method"].lower()=="flame":
        server = FlameServer(params=params_loaded, current_time=current_time, train_dataset=dataloader.train_dataset, 
                            blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                            edge_case_test=dataloader.edge_poison_test)
    elif dataloader.params["defense_method"].lower()=="deepsight":
        server = DeepsightServer(params=params_loaded, current_time=current_time, train_dataset=dataloader.train_dataset, 
                            blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                            edge_case_test=dataloader.edge_poison_test)
    elif dataloader.params["defense_method"].lower()=="foolsgold":
        server = FoolsgoldServer(params=params_loaded, current_time=current_time, train_dataset=dataloader.train_dataset, 
                            blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                            edge_case_test=dataloader.edge_poison_test)

    elif dataloader.params["defense_method"].lower()=="rflbat":
        server = RflbatServer(params=params_loaded, current_time=current_time, train_dataset=dataloader.train_dataset, 
                            blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                            edge_case_test=dataloader.edge_poison_test)
    elif dataloader.params["defense_method"].lower()=="multikrum":
        server = MultikrumServer(params=params_loaded, current_time=current_time, train_dataset=dataloader.train_dataset, 
                            blend_pattern=blend_pattern, edge_case_train=dataloader.edge_poison_train,
                            edge_case_test=dataloader.edge_poison_test)

    if server.params["agg_method"]=="FedProx":
        benign_client = FedProxBenignClient(params=params_loaded, train_dataset=dataloader.train_dataset, 
                                            blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                            edge_case_train=dataloader.edge_poison_train, 
                                            edge_case_test=dataloader.edge_poison_test)
    else:
        benign_client = BenignClient(params=params_loaded, train_dataset=dataloader.train_dataset)

    if server.params["malicious_train_algo"].upper() == "CHAMELEON":
        malicious_client = ChameleonMaliciousClient(params=params_loaded, train_dataset=dataloader.train_dataset, 
                                       blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                       edge_case_train=dataloader.edge_poison_train,
                                       edge_case_test=dataloader.edge_poison_test)
    else:
        malicious_client = MaliciousClient(params=params_loaded, train_dataset=dataloader.train_dataset, 
                                       blend_pattern=blend_pattern, open_set=dataloader.ood_data,
                                       edge_case_train=dataloader.edge_poison_train,
                                       edge_case_test=dataloader.edge_poison_test)

    acc_list = list()
    acc_p_list = list()
    for round in range(server.params["start_round"], server.params["end_round"]):

        server.pre_process(global_data=dataloader.global_data,
                           test_data=dataloader.test_data,
                           round=round
                           )
        if server.params["defense_method"].lower()!="flame":
            weight_accumulator, aggregated_model_id \
                                = server.broadcast_upload(
                                round=round,
                                local_benign_client=benign_client, 
                                local_malicious_client=malicious_client,
                                train_dataloader=dataloader.train_data,
                                poison_train_dataloader=dataloader.poison_data,
                                test_dataloader=dataloader.test_data,
                                global_dataloader=dataloader.global_data
                                )

            server.aggregation(weight_accumulator=weight_accumulator, aggregated_model_id=aggregated_model_id)
        else:
            weight_accumulator, aggregated_model_id, clip_value\
                                = server.broadcast_upload(
                                round=round,
                                local_benign_client=benign_client, 
                                local_malicious_client=malicious_client,
                                train_dataloader=dataloader.train_data,
                                poison_train_dataloader=dataloader.poison_data,
                                test_dataloader=dataloader.test_data,
                                global_dataloader=dataloader.global_data
                                )

            server.aggregation(weight_accumulator=weight_accumulator, aggregated_model_id=aggregated_model_id,
                               clip_value=clip_value)

        logger.info(f" ")
        acc, acc_p = server.global_test(
                        test_data=dataloader.test_data, 
                        round=round, 
                        poisoned_pattern_choose=server.params["poisoned_pattern_choose"]
                        )

        acc_list.append(acc)
        acc_p_list.append(acc_p)
       
        server.post_process()

        save_model(name="global_model", folder_path=server.folder_path, round=round,
                   lr=server.params["benign_lr"], save_on_round=server.params["save_on_round"], 
                   model=server.global_model, ood_dataloader=dataloader.ood_data)
    
    plot_poisoned_acc(save_path=server.folder_path, start_round=server.params["start_round"],
                       acc=acc_list, acc_p=acc_p_list, is_save_img=True)

