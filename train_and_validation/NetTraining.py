import torch
import torch.nn as nn
import torch.optim as optim

#Trains a model under a dataset
def train(model, training_set, validation_set, optimizer, loss_funct, epochs= 30):
    #model.cuda()

    for i in range(epochs):
        train_loss = 0
        validation_loss = 0

        model.train()

        for j, (data, label) in enumerate(training_set):
            data, label = data.cuda(), label.cuda()

            optimizer.zero_grad()

            output = model(data)

            loss = loss_funct(output, label)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()*data.size(0)

        train_loss /= len(training_set.sampler)

        print("Epoch: {} \tTraining loss: {:.6f}".format(i+1, train_loss))
