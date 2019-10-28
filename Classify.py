import torch
import model.AnimalNet as AnimalNet
import model.ObjectNet as ObjectNet
import train_and_validation.DatasetLoader
import train_and_validation.NetTraining
import train_and_validation.NetAccuracy
import torch.nn as nn
import torch.optim as optim
import cv2
import os

def image_input(image):

    image = cv2.imread(image)
    image = cv2.resize(image, (32,32))

    cv2.imshow("animal", image)
    cv2.waitKey(0)
    
    image = torch.tensor(image, dtype= torch.float)
    image = image.view((1,3, 32, 32))

    
    return image


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

train_model = True
# model = ObjectNet.ObjectNet()
model = AnimalNet.AnimalNet()

# model.load_state_dict(torch.load("model_weights_1.pt"))
model.cuda()
if train_model:
    print("Setting up training set, validation set and test set using CIFAR10 dataset")
    train_set, validation_set, test_set = train_and_validation.DatasetLoader.get_dataset('data')
    print("Starting to train model")
    train_and_validation.NetTraining.train(model, train_set, validation_set, optim.SGD(model.parameters(), lr= 0.01), nn.CrossEntropyLoss(), 50)
    print(train_and_validation.NetAccuracy.accuracy(model, test_set))

# accuracy  = 0
# for i in range(12500):
#     print(i, end= '\t')
#     if os.path.exists('Cat_Dog_data/train/dog/dog.{}.jpg'.format(i)):
#         output = model(image_input('Cat_Dog_data/train/dog/dog.{}.jpg'.format(i)).cuda())
#         print(output)
#         print(classes[output.topk(1, dim=1).indices[0]])
#         if classes[output.topk(1, dim=1).indices[0]] == 'dog':
#             accuracy += 1
# print(accuracy/12500 * 100)
