from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from time import time
from USPS import USPS
from torch.utils.data import DataLoader
from torch import nn, optim
import torch


def neural_network():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.3525, 0.3131)])
    train_data = USPS('Data/', train=True, transform=transform, download=True)
    test_data = USPS('Data/', train=False, transform=transform, download=True)
    train_data_loader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    dataiter = iter(train_data_loader)
    (image, label) = dataiter.next()
    # num_of_pixels = len(train_data) * 16 * 16
    # total_sum = 0
    # for batch in train_data_loader: total_sum += batch[0].sum()
    # mean = total_sum / num_of_pixels
    # sum_of_squared_error = 0
    # for batch in train_data_loader: 
    #     sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()
    # std = torch.sqrt(sum_of_squared_error / num_of_pixels)
    # print(mean, std)

    # print(image[0].mean(), image[0].std())
    print(type(image))
    print(type(label))
    print(image.shape)
    print(label.shape)

    input_size = 256
    hidden_sizes = [128, 64]
    output_size = 10

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))
    # print(model)
    criterion = nn.NLLLoss()
    images, labels = next(iter(train_data_loader))
    images = images.view(images.shape[0], -1)

    logps = model(images) #log probabilities
    loss = criterion(logps, labels) #calculate the NLL loss
    # print('Before backward pass: \n', model[0].weight.grad)
    loss.backward()
    # print('After backward pass: \n', model[0].weight.grad)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # print('Initial weights - ', model[0].weight)

    images, labels = next(iter(train_data_loader))
    images.resize_(64, 256)

    # Clear the gradients, do this because gradients are accumulated
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    # print('Gradient -', model[0].weight.grad)

    # Take an update step and few the new weights
    optimizer.step()
    # print('Updated weights - ', model[0].weight)

    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    time0 = time()
    epochs = 15
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_data_loader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(train_data_loader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    images, labels = next(iter(test_data_loader))

    img = images[0].view(1, 256)
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model(img)

    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    view_classify(img.view(1, 16, 16), ps)

    correct_count, all_count = 0, 0
    for images,labels in test_data_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 256)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))

def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 16, 16).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


def main():
    # preprocess_usps()
    neural_network()


if __name__ == '__main__': main()