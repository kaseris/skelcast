from torch.distributed import init_process_group, destroy_process_group

init_process_group(backend='nccl')
destroy_process_group()