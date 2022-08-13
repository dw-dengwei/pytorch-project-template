from util.preprocess import RandomAugment
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from dataset.toy_dataset import SimpleDataset


def create_sampler(datasets, are_shuffle, num_tasks, global_rank):
    samplers = []
    for dataset, is_shuffle in zip(datasets, are_shuffle):
        sampler = DistributedSampler(
            dataset, 
            num_replicas=num_tasks, 
            rank=global_rank, 
            shuffle=is_shuffle
        )
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    

    
def create_dataset(dataset_name, config):
    train_transform = transforms.Compose([])  
    test_transform = transforms.Compose([])  
    train_dataset = SimpleDataset()  
    valid_dataset = SimpleDataset()  
    test_dataset = SimpleDataset()  
    return train_dataset, valid_dataset, test_dataset