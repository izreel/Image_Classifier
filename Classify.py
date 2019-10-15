import torch
import model.ObjectNet as ObjectNet
import train_and_validation.DatasetLoader
import train_and_validation.NetTraining
import train_and_validation.NetAccuracy
import torch.nn as nn
import torch.optim as optim


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

# model.load_state_dict(torch.load("model_weights.pt"))
model.cuda()

print("Setting up training set, validation set and test set using CIFAR10 dataset")
train_set, validation_set, test_set = train_and_validation.DatasetLoader.get_dataset('data')
print("Starting to train model")
train_and_validation.NetTraining.train(model, train_set, validation_set, optim.SGD(model.parameters(), lr= 0.01), nn.CrossEntropyLoss(), 60)
print(train_and_validation.NetAccuracy.accuracy(model, test_set)*100)
