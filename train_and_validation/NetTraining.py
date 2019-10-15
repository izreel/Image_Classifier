import torch
import torch.nn as nn
import torch.optim as optim
import train_and_validation.NetValidation
import numpy as np

#Trains a model under a dataset
def train(model, training_set, validation_set, optimizer, criterion, epochs= 30):
    #model.cuda()
    valid_loss_min = np.Inf
    for i in range(epochs):
        train_loss = 0
        validation_loss = 0

        model.train()

        for data, label in training_set:
            data, label = data.cuda(), label.cuda()

            optimizer.zero_grad()

            output = model(data)

            loss = criterion(output, label)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(training_set.sampler)

        print("Epoch: {} \tTraining loss: {:.6f}".format(i+1, train_loss), end= '')
        
        model.eval()
        with torch.no_grad():
            validation_loss, accuracy = train_and_validation.NetValidation.validate(model, validation_set, criterion )
        
        print("\tTest loss: {:.6f} \tCurrent accuracy: {}".format(validation_loss, accuracy))

        if validation_loss <= valid_loss_min:
            print("Saving weights")
            torch.save(model.state_dict(), "model_weights.pt")
            valid_loss_min = validation_loss
