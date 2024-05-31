import torch.utils.data as data

class AbstractDataloader():
    def __init__(self, params):
        self.params = params
    
    def load_dataset(self):
        r"""
        to load necessary data
        """
        raise NotImplementedError

    def create_loader(self):
        r"""
        create needed dataloader
        """
        raise NotImplementedError
