import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Parse input arguments from the command line
parser = argparse.ArgumentParser(description='CNN for UED',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--target-accuracy', type=float, default=.85,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2,
                    help='Number of epochs that meet target before stopping')

args = parser.parse_args()

# Define a convolutional block followed by batch normalization and ReLU activation
class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        # 7x7 convolution followed by batch normalization and ReLU
        self.cbr = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=(1,1),
                               padding='same', bias=False),
                               nn.BatchNorm2d(output_channels),
                               nn.ReLU()
        )
    def forward(self, x):
        # Forward pass through this block
        out = self.cbr(x)
        return out

# Define a residual convolution block for deeper layers
class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, scale_input):
        super(conv_block, self).__init__()
        self.scale_input = scale_input
        # Downscaling of the input dimensions
        if self.scale_input:
            self.scale = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=(1,1),
                               padding='same')
        # Two consecutive convolutional blocks with dropout
        self.layer1 = cbrblock(input_channels, output_channels)
        self.dropout = nn.Dropout(p=0.1)
        self.layer2 = cbrblock(output_channels, output_channels)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        if self.scale_input:
            residual = self.scale(residual)
        # Add residual
        out = out + residual

        return out

# Define the overall CNN architecture
class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        nChannels = [3, 16, 32, 64, 128]  # Channel sizes
        self.input_block = cbrblock(nChannels[0], nChannels[1])  # Initial convolutional block
        self.block1 = conv_block(nChannels[1], nChannels[2], 1)  # Residual blocks
        self.block2 = conv_block(nChannels[2], nChannels[2], 0)
        self.pool1 = nn.MaxPool2d(9)
        self.block3 = conv_block(nChannels[2], nChannels[3], 1)
        self.block4 = conv_block(nChannels[3], nChannels[3], 0)
        self.pool2 = nn.MaxPool2d(9)
        self.block5 = conv_block(nChannels[3], nChannels[4], 1)
        self.block6 = conv_block(nChannels[4], nChannels[4], 0)
        self.pool = nn.AvgPool2d(3)  # Global average pooling
        self.flat = nn.Flatten()  # Flatten the output
        self.fc = nn.Linear(nChannels[4], num_classes)  # Fully connected output layer

    def forward(self, x):
        # Pass input through all layers
        out = self.input_block(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = self.flat(out)
        out = self.fc(out)

        return out

# Function for training the model
def train(model, optimizer, train_loader, loss_fn, device):
    model.train()
    for images, labels in train_loader:
        labels = labels.to(device)
        images = images.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Zero gradients before backward pass
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

# Function for testing the model
def test(model, test_loader, loss_fn, device):
    total_labels = 0
    correct_labels = 0
    loss_total = 0
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            predictions = torch.max(outputs, 1)[1]
            total_labels += len(labels)
            correct_labels += (predictions == labels).sum()
            loss_total += loss
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)

    # Calculate and plot the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix for CNN')
    plt.savefig('/scratch/user/mohammad.shaaban/VENV/confusion_matrix.png')
    plt.close()

    return v_accuracy, v_loss

# Main training loop
if __name__ == '__main__':
    train_set = torchvision.datasets.ImageFolder('/scratch/user/mohammad.shaaban/VENV/CNN_70k_images', 
                                                 transform=transforms.Compose([transforms.ToTensor()]))
    test_set = torchvision.datasets.ImageFolder('/scratch/user/mohammad.shaaban/VENV/CNN_7k_images', 
                                                transform=transforms.Compose([transforms.ToTensor()]))

    # Ensure subsets do not exceed the actual dataset sizes
    train_subset_size = min(70000, len(train_set))
    test_subset_size = min(7000, len(test_set))

    # Create data loaders
    train_subset = torch.utils.data.Subset(train_set, list(range(train_subset_size)))
    test_subset = torch.utils.data.Subset(test_set, list(range(test_subset_size)))

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=args.batch_size, drop_last=True, shuffle=True)

    num_classes = 7  # Define the number of classes for classification
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = WideResNet(num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()  # Loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.5)  # Optimizer

    total_time = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        train(model, optimizer, train_loader, loss_fn, device)
        epoch_time = time.time() - t0
        images_per_sec = len(train_subset)/epoch_time
        v_accuracy, v_loss = test(model, test_loader, loss_fn, device)
        print("Epoch = {:2d}: Epoch Time = {:5.3f}, Validation Loss = {:5.3f}, Validation Accuracy = {:5.3f}, Image throughput = {:7.3f} images/sec".format(epoch+1, epoch_time, v_loss, v_accuracy, images_per_sec))
        if epoch >= args.patience - 1:
            recent_accuracies = [x.cpu() for x in val_accuracy[-args.patience:]]
            if all(acc > args.target_accuracy for acc in recent_accuracies):
                print('Early stopping after epoch {}'.format(epoch + 1))
                break
        torch.save(model.state_dict(), '/scratch/user/mohammad.shaaban/VENV/Model_BS-lr_0.01-256')    
        total_time += epoch_time
    print("Total time = {:8.3f} seconds".format(total_time))
