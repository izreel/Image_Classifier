import numpy as np
import torch

def accuracy(model, test_set):
    acc = 0

    for images, labels in test_set:
        images, labels = images.cuda(), labels.cuda()
        output = (model(images))

        for i in range(images.shape[0]):
            if output.topk(1, dim=1).indices[i] == labels[i]:
                acc += 1

    return acc/len(test_set.sampler)
