import torch
import model.ObjectNet as ObjectNet
import train_and_validation.DatasetLoader

#class labels in CIFAR-10 dataset
classes = ['airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck']

model = ObjectNet.ObjectNet()

model = model.cuda()
train_set, validation_set, test_set = train_and_validation.DatasetLoader.get_dataset('data')