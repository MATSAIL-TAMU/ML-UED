import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image
import numpy as np

# Parse input arguments
parser = argparse.ArgumentParser(description='VAE for UED', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--latent-dim', type=int, default=4, help='Dimension of the latent space')
parser.add_argument('--base-lr', type=float, default=0.00015, help='learning rate for a single GPU')
parser.add_argument('--data-path', default='/scratch/user/mohammad.shaaban/VENV/UED_images', help='Path for training data image directory')
parser.add_argument('--save-path', default='/scratch/user/mohammad.shaaban/VENV/trained_model.pth', help='Path for saving trained model')

args = parser.parse_args()

# Hyperparameters for the VAE
rotate_images = False
latent_dim = args.latent_dim
hidden_dim = 2 * latent_dim  # Hidden dimension in the encoder/decoder
input_channels = 3  # Number of input channels (e.g., RGB images)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading and augmentation
if rotate_images:
    image_transform = transforms.Compose([transforms.Resize((261, 261)), transforms.RandomRotation(degrees=(-60, 60)), transforms.ToTensor()])
else:
    image_transform = transforms.Compose([transforms.Resize((261, 261)), transforms.ToTensor()])

train_dataset = ImageFolder(args.data_path, transform=image_transform)

kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

# Encoder module for VAE
class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        # Series of convolutional layers to downsample and extract features
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, 4, 2, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )
        # Calculate the size of the flattened feature map after convolutions
        self.flatten_size = self._get_flatten_size(input_channels)
        # Fully connected layers for generating latent mean and log variance
        self.fc = nn.Linear(self.flatten_size, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def _get_flatten_size(self, input_channels):
        # Calculate the size of the output after all the convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 261, 261)
            dummy_output = self.features(dummy_input)
            return dummy_output.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)  # Extract features through convolutions
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.fc(x)  # Project to hidden dimension
        mean = self.fc_mean(x)  # Compute mean of the latent distribution
        logvar = self.fc_logvar(x)  # Compute log variance of the latent distribution
        return mean, logvar

# Decoder module for VAE
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_channels):
        super(Decoder, self).__init__()
        # Project latent space back into a large feature map
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2048 * 1 * 1)  # Adjusted to match encoder's output size
        # Transposed convolutional layers to reconstruct the image
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, 4, 2, 1),
            nn.Sigmoid(),  # Output in the range [0,1] for image reconstruction
        )

    def forward(self, z):
        z = self.fc(z)
        z = self.fc2(z)
        z = z.view(z.size(0), 2048, 1, 1)  # Reshape back into feature maps
        z = self.layers(z)  # Decode to reconstruct the image
        return z

# Complete VAE model with encoder and decoder
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_channels)

    # Reparameterization trick: generates latent variable z from mean and logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Compute standard deviation
        eps = torch.randn_like(std)  # Sample random noise
        return mu + eps * std  # Return the reparameterized latent variable

    def forward(self, x):
        mu, logvar = self.encoder(x)  # Get mean and logvar from encoder
        z = self.reparameterize(mu, logvar)  # Generate latent variable
        return self.decoder(z), mu, logvar  # Reconstruct the image and return latent vars

# Loss function for VAE
def loss_function(recon_x, x, mu, logvar):
    # Resize the input to match the reconstructed output dimensions
    x_resized = nn.functional.interpolate(x, size=(recon_x.size(2), recon_x.size(3)), mode='bilinear', align_corners=False)

    # Reconstruction loss (Binary Cross-Entropy) and KL divergence loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x_resized, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD  # Total loss

# Initialize class means for each class in the dataset
class_means_sums = {i: torch.zeros(latent_dim).to(DEVICE) for i in range(len(train_dataset.classes))}
class_counts = {i: 0 for i in range(len(train_dataset.classes))}

# Save class mean vectors to a file
def save_class_means(average_means):
    with open("/scratch/user/mohammad.shaaban/VENV/class_means_0715_full.txt", "w") as f:
        for class_id, means in average_means.items():
            means_list = means.cpu().detach().numpy().tolist()
            means_str = ','.join([f'{value:.8f}' for value in means_list])
            f.write(f"Class {class_id}: [{means_str}]\n")

# Generate and save images from the average class means
def generate_and_save_images(decoder, average_means):
    for class_id, means in average_means.items():
        z = means.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            generated_image = decoder(z)  # Generate image from mean
        save_image(generated_image, f"/scratch/user/mohammad.shaaban/VENV/average_image_class_{class_id}.png")

# Generate interpolated images between two classes
def generate_specific_interpolated_images(decoder, average_means, num_interpolations=3):
    start_mean = average_means[0]  # Mean of class 0
    end_mean = average_means[1]  # Mean of class 1
    for interp in range(num_interpolations + 2):  # Interpolate between means
        fraction = interp / (num_interpolations + 1)
        interpolated_mean = start_mean * (1 - fraction) + end_mean * fraction
        z = interpolated_mean.unsqueeze(0)
        with torch.no_grad():
            generated_image = decoder(z)  # Generate interpolated image
        save_image(generated_image, f"/scratch/user/mohammad.shaaban/VENV/interpolated_image_class_0_to_1_{interp}.png")

# Training function for VAE
def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    all_mu = []  # Store the latent means for all samples
    all_labels = []  # Store corresponding labels
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)  # Ensure targets are moved to the correct device
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)  # Forward pass through VAE
        
        loss = loss_function(recon_batch, data, mu, logvar)  # Compute loss
        loss.backward()  # Backpropagate
        train_loss += loss.item()
        optimizer.step()  # Update weights

        # Update class mean sums for calculating average means
        for i in range(data.size(0)):
            class_means_sums[targets[i].item()] += mu[i]
            class_counts[targets[i].item()] += 1

        all_mu.append(mu.detach().cpu())  # Store latent mean for this batch
        all_labels.append(targets.detach().cpu())  # Store corresponding labels

    print(f'Epoch: {epoch}, Average loss: {train_loss / len(train_loader.dataset)}')
    return torch.cat(all_mu), torch.cat(all_labels)  # Return all latent means and labels

# Initialize the VAE model and optimizer
model = VAE().to(DEVICE)
optimizer = Adam(model.parameters(), lr=args.base_lr)

# Main training loop
all_mu_epoch = []
all_labels_epoch = []
for epoch in range(args.epochs):
    mu_epoch, labels_epoch = train(model, train_loader, optimizer, epoch)
    all_mu_epoch.append(mu_epoch)
    all_labels_epoch.append(labels_epoch)

    if epoch % 50 == 0:  # Save model checkpoint every 50 epochs
        temp_savefile = 'iter_' + str(epoch) + '_' + args.save_path
        temp_dir = os.path.dirname(temp_savefile)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        torch.save(model.state_dict(), temp_savefile)
        torch.save(model.state_dict(), args.save_path)

# Save final model state
torch.save(model.state_dict(), args.save_path)

# Calculate and save the average class means
average_means = {class_id: class_means_sums[class_id] / class_counts[class_id] for class_id in class_means_sums}
save_class_means(average_means)

# Generate and save average class images
generate_and_save_images(model.decoder, average_means)

# Generate interpolated images between classes
generate_specific_interpolated_images(model.decoder, average_means, num_interpolations=3)
