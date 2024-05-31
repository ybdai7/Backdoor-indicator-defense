import os
import torch
import logging
import yaml
import models.resnet
import models.vgg

logger = logging.getLogger("logger")

class AbstractServer():
    r"""
    AbstractServer is an abstract class in which the aggregation(), broadcast_upload(), 
    global_benign_test(), global_poison_test() and etc. should be implemented according
    to specific strategies.
    """
    
    def __init__(self, params, current_time):
        self.params = params
        self.current_time = current_time
        self.folder_path = f"./saved_models/{self.current_time}"
        try:
            os.makedirs(self.folder_path)
        except FileExistsError:
            logger.info("Folder already exists")
        logger.addHandler(logging.FileHandler(filename=f"{self.folder_path}/log.txt"))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        logger.info(f"current path:{self.folder_path}")

        self._create_model()
        self._resume_model()
        
        self.poisoned_rounds = [round for round in range(self.params["poisoned_start_round"],
                                                         self.params["poisoned_end_round"],
                                                         self.params["poisoned_round_interval"])] 

        # logger.info(f"poisoned rounds:{self.poisoned_rounds}")

        with open(f"{self.folder_path}/params.yaml", "w") as f:
            yaml.dump(self.params, f, sort_keys=False)

    def _create_model(self):
        r"""
        create global model according to the uploaded params info,
        ATTENTION: VGG model does not support EMNIST task YET!
        """
        if "ResNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                global_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="CIFAR")
            elif self.params["dataset"].upper() == "CIFAR100":
                global_model = getattr(models.resnet, self.params["model_type"])(num_classes=100, dataset="CIFAR")
            elif self.params["dataset"].upper() == "EMNIST":
                global_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="EMNIST")

        elif "VGG" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                global_model = getattr(models.vgg, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                global_model = getattr(models.vgg, self.params["model_type"])(num_classes=100)

        
        self.global_model = global_model.cuda()
        
        return True

    def _resume_model(self):
        r"""
        resume model from checkpoint
        """
        if self.params["resumed_model"]:
            # path=resume_last("saved_models")
            loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
            self.global_model.load_state_dict(loaded_params["state_dict"])
            self.params["start_round"] = loaded_params["round"]
            # self.params["benign_lr"] = loaded_params.get("lr", self.params["benign_lr"])
            logger.info(f"Loaded params from saved model, LR is {self.params['benign_lr']} and current round is {self.params['start_round']}")
        else:
            self.params["start_round"] = 1
            logger.info(f"start training from the 1st round")

        return True

    def _select_client(self):
        r"""
        to select participants for every round
        """
        raise NotImplementedError

    def aggregation(self, weight_accumulator):
        r"""
        to aggregate the local updates to generate new global
        model
        """
        raise NotImplementedError

    def broadcast_upload(self, local_client):
        r"""
        the server broadcasts global model at t round, local clients train the global model
        based on the received model and upload to the server after training.
        This function return the weight_accumulator of all the participants' uploaded model.
        """
        raise NotImplementedError

    def _global_test_sub(self, test_data):
        r"""
        to test the sub task accuracy on global model
        """
        raise NotImplementedError
    
    def global_test(self, test_data):
        r"""
        to test different tasks' accuracy on global model
        """
        raise NotImplementedError
   
    def post_process(self):
        r"""
        post process by server after aggregation
        """
        raise NotImplementedError
