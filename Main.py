import os
import csv
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image

# ---------------------------
#       Hyperparameters
# ---------------------------
batch_size = 128
learning_rate = 0.0002
num_epochs = 100
z_dim = 100
num_classes = 100  # CIFAR100 has 100 classes
img_size = 32      # CIFAR100 image size
img_channels = 3   # RGB images

# ---------------------------
#         Paths & Device
# ---------------------------
base_results_path = 'results'
all_results_path = os.path.join(base_results_path, 'all_hyperparameters.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------
#         Seeding
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ---------------------------
#       Model Components
# ---------------------------

# ResNet Block for skip connections
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x  # Skip connection
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, width, height = x.size()
        query = self.query(x).view(batch, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, width * height)
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)
        value = self.value(x).view(batch, -1, width * height)
        attention = torch.bmm(value, attention.permute(0, 2, 1))
        attention = attention.view(batch, C, width, height)
        out = self.gamma * attention + x
        return out

# ---------------------------
#      Generator Models
# ---------------------------

class ResNetGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=100, img_channels=3, feature_maps=64):
        super(ResNetGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + num_classes, feature_maps * 8, 4, 1, 0, bias=False),
            ResNetBlock(feature_maps * 8, feature_maps * 8),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            ResNetBlock(feature_maps * 4, feature_maps * 4),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            ResNetBlock(feature_maps * 2, feature_maps * 2),
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, label_embedding], dim=1)
        return self.model(x)


class SkipConnectionGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=100, img_channels=3, feature_maps=64):
        super(SkipConnectionGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.initial = nn.ConvTranspose2d(z_dim + num_classes, feature_maps * 8, 4, 1, 0, bias=False)
        self.channel_matcher = nn.Conv2d(feature_maps * 8, img_channels, kernel_size=1)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # Define sharpening kernel for skip connection
        sharpen_kernel = torch.tensor(
            [[[-1., -1., -1.],
              [-1.,  9., -1.],
              [-1., -1., -1.]]],
            dtype=torch.float32
        )
        self.register_buffer('sharpen_kernel', sharpen_kernel.repeat(img_channels, 1, 1, 1))

    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, label_embedding], dim=1)
        skip = self.initial(x)
        out = self.model(skip)
        skip_upsampled = F.interpolate(skip, size=(out.size(2), out.size(3)), mode='bilinear', align_corners=False)
        skip_projected = self.channel_matcher(skip_upsampled)
        skip_sharpened = F.conv2d(skip_projected, self.sharpen_kernel, padding=1, groups=img_channels)
        combined = out + skip_sharpened
        return combined


class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=100, img_channels=3, feature_maps=64):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + num_classes, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, label_embedding], dim=1)
        return self.model(x)


class ImprovedGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=100, img_channels=3, feature_maps=64):
        super(ImprovedGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + num_classes, feature_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            SelfAttention(feature_maps * 8),  # Self-attention layer for better global feature modeling
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, label_embedding], dim=1)
        return self.model(x)


class ESkipConnectionGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=100, img_channels=3, feature_maps=64):
        super(ESkipConnectionGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.initial = nn.ConvTranspose2d(z_dim + num_classes, feature_maps * 8, 4, 1, 0, bias=False)
        self.channel_matcher = nn.Conv2d(feature_maps * 8, img_channels, kernel_size=1)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        # Sharpening kernel for enhanced skip connection
        sharpen_kernel = torch.tensor(
            [[[-1., -1., -1.],
              [-1.,  9., -1.],
              [-1., -1., -1.]]],
            dtype=torch.float32
        )
        self.register_buffer('sharpen_kernel', sharpen_kernel.repeat(img_channels, 1, 1, 1))

    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, label_embedding], dim=1)
        skip = self.initial(x)
        out = self.model(skip)
        skip_upsampled = F.interpolate(skip, size=(out.size(2), out.size(3)), mode='bilinear', align_corners=False)
        skip_projected = self.channel_matcher(skip_upsampled)
        skip_sharpened = F.conv2d(skip_projected, self.sharpen_kernel, padding=1, groups=img_channels)
        combined = 0.5 * out + 0.5 * skip_sharpened  # Weighted combination
        return combined

# ---------------------------
#      Discriminator Models
# ---------------------------

class ResNetDiscriminator(nn.Module):
    def __init__(self, num_classes=100, img_channels=3, feature_maps=64):
        super(ResNetDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, img_channels * 32 * 32)
        self.model = nn.Sequential(
            nn.Conv2d(img_channels * 2, feature_maps, 4, 2, 1, bias=False),
            ResNetBlock(feature_maps, feature_maps),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            ResNetBlock(feature_maps * 2, feature_maps * 2),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            ResNetBlock(feature_maps * 4, feature_maps * 4),
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(images.size(0), img_channels, 32, 32)
        x = torch.cat([images, label_embedding], dim=1)
        return self.model(x)


class SkipConnectionDiscriminator(nn.Module):
    def __init__(self, num_classes=100, img_channels=3, feature_maps=64):
        super(SkipConnectionDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, img_channels * 32 * 32)
        self.initial_conv = nn.Conv2d(img_channels * 2, feature_maps, 4, 2, 1, bias=False)
        self.model = nn.Sequential(
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.skip_projection = nn.Conv2d(img_channels * 2, feature_maps * 8, kernel_size=1, stride=1, bias=False)
        edge_kernel = torch.tensor(
            [[[-1., -1., -1.],
              [-1.,  8., -1.],
              [-1., -1., -1.]]],
            dtype=torch.float32
        )
        self.register_buffer('edge_kernel', edge_kernel.repeat(feature_maps * 8, 1, 1, 1))

    def forward(self, images, labels):
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(images.size(0), images.size(1), 32, 32)
        x = torch.cat([images, label_embedding], dim=1)
        skip_connection = x.clone()
        x = self.initial_conv(x)
        x = self.model(x)
        skip_connection_upsampled = F.interpolate(skip_connection, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        skip_connection_projected = self.skip_projection(skip_connection_upsampled)
        skip_edge_detected = F.conv2d(skip_connection_projected, self.edge_kernel, padding=1, groups=self.skip_projection.out_channels)
        x = x + skip_edge_detected
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=100, img_channels=3, feature_maps=64):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, img_channels * 32 * 32)
        self.model = nn.Sequential(
            nn.Conv2d(img_channels * 2, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(images.size(0), img_channels, 32, 32)
        x = torch.cat([images, label_embedding], dim=1)
        return self.model(x)


class ImprovedDiscriminator(nn.Module):
    def __init__(self, num_classes=100, img_channels=3, feature_maps=64):
        super(ImprovedDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, img_channels * 64 * 64)
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_channels * 2, feature_maps, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, images, labels):
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(images.size(0), images.size(1), 64, 64)
        label_embedding = F.interpolate(label_embedding, size=(images.size(2), images.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat([images, label_embedding], dim=1)
        x = self.model(x)
        x = self.fc(x)
        return x


class ESkipConnectionDiscriminator(nn.Module):
    def __init__(self, num_classes=100, img_channels=3, feature_maps=64):
        super(ESkipConnectionDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, img_channels * 32 * 32)
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(img_channels * 2, feature_maps, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 8, 1, 2, 1, 0, bias=False)
        )
        self.skip_conv = nn.Conv2d(img_channels * 2, feature_maps * 8, kernel_size=1, stride=1, bias=False)
        edge_kernel = torch.tensor(
            [[[-1., -1., -1.],
              [-1.,  8., -1.],
              [-1., -1., -1.]]],
            dtype=torch.float32
        )
        self.register_buffer('edge_kernel', edge_kernel.repeat(feature_maps * 8, 1, 1, 1))

    def forward(self, images, labels):
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(images.size(0), images.size(1), 32, 32)
        x = torch.cat([images, label_embedding], dim=1)
        main_out = self.model(x)
        skip_projected = self.skip_conv(x)
        skip_upsampled = F.interpolate(skip_projected, size=(main_out.size(2), main_out.size(3)), mode='bilinear', align_corners=False)
        skip_edge_detected = F.conv2d(skip_upsampled, self.edge_kernel, padding=1, groups=self.skip_conv.out_channels)
        combined = main_out + skip_edge_detected
        return combined

# ---------------------------
#       Data Preparation
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def create_folders():
    os.makedirs(base_results_path, exist_ok=True)

def create_experiment_folder(experiment_name):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_path = os.path.join(base_results_path, f'{experiment_name}_{timestamp}')
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(os.path.join(experiment_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(experiment_path, 'models'), exist_ok=True)
    return experiment_path

def log_epoch_data(experiment_path, epoch, epoch_metrics):
    epoch_results_path = os.path.join(experiment_path, 'epoch_data.csv')
    if not os.path.exists(epoch_results_path):
        with open(epoch_results_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=epoch_metrics.keys())
            writer.writeheader()
    with open(epoch_results_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=epoch_metrics.keys())
        writer.writerow(epoch_metrics)

def normalize_block(block):
    return nn.Sequential(
        block,
        nn.BatchNorm2d(block.out_channels),
        nn.ReLU(inplace=True)
    )

# ---------------------------
#        Training Function
# ---------------------------
def train_gan(generator_class, discriminator_class, experiment_name):
    experiment_path = create_experiment_folder(experiment_name)
    
    generator = generator_class(z_dim=z_dim, num_classes=num_classes, img_channels=img_channels).to(device)
    discriminator = discriminator_class(num_classes=num_classes, img_channels=img_channels).to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        print(f"\n[INFO] Starting Epoch {epoch+1}/{num_epochs} for {experiment_name}")
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss_D_real': 0.0,
            'loss_D_fake': 0.0,
            'loss_D_total': 0.0,
            'loss_G': 0.0,
            'combined_loss': 0.0
        }
        
        generator.train()
        discriminator.train()
        
        for i, (real_images, labels) in enumerate(dataloader):
            real_images, labels = real_images.to(device), labels.to(device)
            current_batch_size = real_images.size(0)
            
            # Create label tensors for real and fake
            real_labels = torch.ones(current_batch_size, device=device)
            fake_labels = torch.zeros(current_batch_size, device=device)
            
            # Generate fake images
            noise = torch.randn(current_batch_size, z_dim, 1, 1, device=device)
            fake_images = generator(noise, labels)
            fake_images = F.interpolate(fake_images, size=(real_images.size(2), real_images.size(3)),
                                          mode='bilinear', align_corners=True)

            # --------------------
            # Train Discriminator
            # --------------------
            discriminator.zero_grad()
            real_output = discriminator(real_images, labels).view(-1)
            loss_real = -real_output.mean()
            fake_output = discriminator(fake_images.detach(), labels).view(-1)
            loss_fake = fake_output.mean()

            # Gradient Penalty
            alpha = torch.rand(current_batch_size, 1, 1, 1, device=device)
            interpolated_images = alpha * real_images + (1 - alpha) * fake_images
            interpolated_images.requires_grad_(True)
            interpolated_scores = discriminator(interpolated_images, labels)
            gradients = torch.autograd.grad(
                outputs=interpolated_scores,
                inputs=interpolated_images,
                grad_outputs=torch.ones_like(interpolated_scores),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
            gradient_penalty = ((grad_norm - 1) ** 2).mean()

            loss_D = loss_fake - loss_real + 10 * gradient_penalty
            loss_D.backward()
            optimizer_D.step()

            # --------------------
            # Train Generator
            # --------------------
            generator.zero_grad()
            noise = torch.randn(current_batch_size, z_dim, 1, 1, device=device)
            fake_images = generator(noise, labels)
            fake_images = F.interpolate(fake_images, size=(real_images.size(2), real_images.size(3)),
                                        mode='bilinear', align_corners=True)
            output = discriminator(fake_images, labels).view(-1)
            loss_G = -output.mean()
            loss_G.backward()
            optimizer_G.step()

            # Update metrics
            epoch_metrics['loss_D_real'] += loss_real.item()
            epoch_metrics['loss_D_fake'] += loss_fake.item()
            epoch_metrics['loss_D_total'] += loss_D.item()
            epoch_metrics['loss_G'] += loss_G.item()
            epoch_metrics['combined_loss'] += (loss_D.item() + loss_G.item())

        total_batches = len(dataloader)
        for key in ['loss_D_real', 'loss_D_fake', 'loss_D_total', 'loss_G', 'combined_loss']:
            epoch_metrics[key] /= total_batches

        print(f"[INFO] Epoch {epoch+1} Complete - "
              f"Loss D Real: {epoch_metrics['loss_D_real']:.4f}, "
              f"Loss D Fake: {epoch_metrics['loss_D_fake']:.4f}, "
              f"Loss D Total: {epoch_metrics['loss_D_total']:.4f}, "
              f"Loss G: {epoch_metrics['loss_G']:.4f}, "
              f"Combined Loss: {epoch_metrics['combined_loss']:.4f}")

        log_epoch_data(experiment_path, epoch, epoch_metrics)
        torch.save(generator.state_dict(), os.path.join(experiment_path, 'models', f'generator_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(experiment_path, 'models', f'discriminator_epoch_{epoch+1}.pth'))

        # Save example images for a few random classes
        generator.eval()
        with torch.no_grad():
            random_classes = random.sample(range(num_classes), 3)
            for class_idx in random_classes:
                fixed_noise = torch.randn(1, z_dim, 1, 1, device=device)
                fake_image = generator(fixed_noise, torch.tensor([class_idx], device=device))
                fake_image = (fake_image + 1) / 2  # Rescale to [0, 1]
                save_image(fake_image, os.path.join(experiment_path, 'images', f'epoch_{epoch+1}_fake_class_{class_idx}.png'))
            
            real_images, real_labels = next(iter(dataloader))
            for class_idx in random_classes:
                matching_indices = (real_labels == class_idx).nonzero(as_tuple=True)[0]
                if matching_indices.numel() > 0:
                    real_image = real_images[matching_indices[0]].unsqueeze(0).to(device)
                    real_image = (real_image + 1) / 2  # Rescale to [0, 1]
                    save_image(real_image, os.path.join(experiment_path, 'images', f'epoch_{epoch+1}_real_class_{class_idx}.png'))

    return epoch_metrics['loss_G']

# ---------------------------
#            Main
# ---------------------------
def main():
    create_folders()
    train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    global dataloader
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # List of experiments to run (uncomment the ones you want to test)
    experiments = [
        # (ConditionalGenerator, ConditionalDiscriminator),
        # (ConditionalGenerator, SkipConnectionDiscriminator),
        # (ConditionalGenerator, ResNetDiscriminator),
        # (SkipConnectionGenerator, ConditionalDiscriminator),
        # (ImprovedGenerator, ImprovedDiscriminator),
        (ESkipConnectionGenerator, ESkipConnectionDiscriminator),
        # (SkipConnectionGenerator, ResNetDiscriminator),
        # (ResNetGenerator, ConditionalDiscriminator),
        # (ResNetGenerator, SkipConnectionDiscriminator),
        # (ResNetGenerator, ResNetDiscriminator)
    ]

    for gen_class, disc_class in experiments:
        experiment_name = f'{gen_class.__name__}_{disc_class.__name__}'
        print(f'\n[INFO] Starting experiment: {experiment_name}')
        train_gan(gen_class, disc_class, experiment_name)

if __name__ == "__main__":
    main()
