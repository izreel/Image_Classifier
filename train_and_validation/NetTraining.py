import torch
import torch.nn as nn
import torch.optim as optim
import train_and_validation.NetValidation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Trains a model under a dataset
def train(model, training_set, validation_set, optimizer, criterion, epochs= 30):

    valid_loss_min = np.Inf
    losses = pd.DataFrame(columns= ['train', 'valid', 'accuracy'])
    losses.loc[0] = [0,0,0]
    for i in range(epochs):
        print("Epoch: {}\t".format(i+1), end= '')
        
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

        print("Training loss: {:.6f}\t".format(train_loss), end= '')
        
        model.eval()
        with torch.no_grad():
            validation_loss, accuracy = train_and_validation.NetValidation.validate(model, validation_set, criterion )
            losses.loc[len(losses)] = [train_loss, validation_loss, accuracy]
        
        print("Test loss: {:.6f}\tCurrent accuracy: {}".format(validation_loss, accuracy))

        if validation_loss <= valid_loss_min:
            print("Saving weights")
            torch.save(model.state_dict(), "{}_model_weights_{}_epochs.pt".format(model.__class__.__name__, epochs))
            valid_loss_min = validation_loss
  
    losses[["train", "valid"]].plot()
    plt.xlabel('Epoch')
    plt.ylabel("Losses")
    plt.title('Training and Validation Results through each Epoch')
    plt.savefig('{}_training_progress_{}_epochs.png'.format(model.__class__.__name__, epochs))

