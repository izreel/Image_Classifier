import train_and_validation.NetAccuracy

def validate(model, validation_set, criterion):
    validation_loss = 0
    accuracy = train_and_validation.NetAccuracy.accuracy(model, validation_set)

    for images, labels in validation_set:
        images, labels = images.cuda(), labels.cuda()

        output = model(images)
        validation_loss += criterion(output, labels).item()
        


    return validation_loss/len(validation_set.sampler), accuracy
