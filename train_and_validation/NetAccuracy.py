import numpy as np
import torch

def accuracy(model, test_set):
    overall_accuracy = 0
    classes_accuracy = np.zeros((10,1))
    for images, labels in test_set:
        images, labels = images.cuda(), labels.cuda()
        output = (model(images))

        for i in range(images.shape[0]):
            if output.topk(1, dim=1).indices[i] == labels[i]:
                overall_accuracy += 1
                classes_accuracy[labels[i]] += 1
    # print(classes_accuracy/1000 * 100)
    return overall_accuracy/len(test_set.sampler) * 100
