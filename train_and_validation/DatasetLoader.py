import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

#Loads CIFAR-10 dataset and augments it (changing the images of the dataset) to be used for training and testing
def get_dataset(dataset_location, batch_size = 20, validation_size = 0.2):
    dataset_transform = transforms.Compose([ transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5))])
    
    train_data = datasets.CIFAR10(dataset_location, train= True, download= False, transform= dataset_transform)
    test_data = datasets.CIFAR10(dataset_location, train= False, download= False, transform= dataset_transform)

    train_indices = list(range(len(train_data)))
    np.random.shuffle(train_indices)
    dataset_split = int(np.floor(validation_size * len(train_data)))

    train_indices, validation_indices = train_indices[dataset_split:], train_indices[:dataset_split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, sampler= train_sampler)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size, sampler= validation_sampler)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= batch_size)
    

    return train_loader, valid_loader, test_loader
