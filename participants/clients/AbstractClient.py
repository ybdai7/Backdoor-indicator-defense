import models.resnet
import models.vgg

class AbstractClient():
    r"""
    AbstractClient is an abstract class in which the local_training(), local_benign_test() and local_poison_test() should be implemented according
    to specific strategies.
    """
    def __init__(self, params):
        self.params = params
        self._create_model()

    def _create_model(self):
        r"""
        create global model according to the uploaded params info,
        ATTENTION: VGG model does not support EMNIST task YET!
        """
        if "ResNet" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                local_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="CIFAR")
            elif self.params["dataset"].upper() == "CIFAR100":
                local_model = getattr(models.resnet, self.params["model_type"])(num_classes=100, dataset="CIFAR")
            elif self.params["dataset"].upper() == "EMNIST":
                local_model = getattr(models.resnet, self.params["model_type"])(num_classes=10, dataset="EMNIST")

        elif "VGG" in self.params["model_type"]:
            if self.params["dataset"].upper() == "CIFAR10":
                local_model = getattr(models.vgg, self.params["model_type"])(num_classes=10)
            elif self.params["dataset"].upper() == "CIFAR100":
                local_model = getattr(models.vgg, self.params["model_type"])(num_classes=100)

        
        self.local_model = local_model.cuda()
        
        return True

    def _optimizer(self):
        r"""
        create optimizer used for training
        """
        raise NotImplementedError

    def _loss_function(self):
        r"""
        create loss function used for training
        """
        raise NotImplementedError

    def local_training(self, train_data):
        r"""
        local training process for client
        """
        raise NotImplementedError
    
    def _local_test_sub(self, test_data):
        r"""
        local test process for subtask of local client
        """
        raise NotImplementedError

    def local_test(self, test_data):
        r"""
        local test task
        """
        raise NotImplementedError
