import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

from skelcast.models import SkelcastModule
from skelcast.experiments import RUNNERS

@RUNNERS.register_module()
class DistributedRunner:
    
    def __init__(self,
                 train_set: Dataset,
                 val_set: Dataset,
                 train_batch_size: int,
                 val_batch_size: int,
                 block_size: int,
                 model: SkelcastModule,
                 optimizer: torch.optim.Optimizer = None,) -> None:
        
        self.train_set = train_set
        self.val_set = val_set
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.block_size = block_size
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.train_sampler = DistributedSampler(self.train_set)
        self.val_sampler = DistributedSampler(self.val_set)
        self.train_loader = DataLoader(self.train_set, batch_size=self.train_batch_size, sampler=self.train_sampler)
        self.val_loader = DataLoader(self.val_set, batch_size=self.val_batch_size, sampler=self.val_sampler)
        self.model = DistributedDataParallel(self.model, device_ids=[self.device])
        